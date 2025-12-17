from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import CategoricalFeatureVectorizer, build_text_encoder
from .mapping import ClusterModelMapper
from .branches.limbo_branch import LimboBranch
from .branches.bert_branch import BertBranch
from .models.fusion_torch import MLPFuserTorch, AttentionFuserTorch, train_fuser
from .features import QueryFeatureExtractor
from limbo_cluster import LimboAgglomerative  
from .logger import RunLogger, RunInfo
from .core import TrainConfig
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


class RouterPipeline:
    """
    - LIMBO 
    - BERT
    - 融合网络
    """

    def __init__(self, model_names: List[str], model_costs: List[float], config: TrainConfig) -> None:
        self.model_names = list(model_names)
        self.model_costs = list(model_costs)
        self.config = config

        # 分支模块
        limbo_coarse_clusters = getattr(config, "limbo_coarse_clusters", None)
        limbo_fit_size = getattr(config, "limbo_fit_size", 2000)
        limbo_prob_temp = getattr(config, "limbo_prob_temp", 1.0)
        limbo_random_state = getattr(config, "seed", 42)
        self.limbo_branch = LimboBranch(
            model_names,
            model_costs,
            config.num_clusters,
            # 传入用于 utility 的 beta（兼容：config.utility_beta 默认为 legacy beta）
            getattr(config, "utility_beta", config.beta),
            config.objective,
            config.quality_threshold,
            limbo_tau=getattr(config, "limbo_tau", None),
            limbo_use_sparse=getattr(config, "limbo_use_sparse", True),
            input_dump_path=getattr(config, "limbo_input_dump_path", None),
            coarse_clusters=limbo_coarse_clusters,
            limbo_fit_size=limbo_fit_size,
            prob_temp=limbo_prob_temp,
            random_state=limbo_random_state,
            progress_logging=getattr(config, "show_progress", True),
        )
        self.bert_branch = BertBranch(
            model_names,
            model_costs,
            config.num_clusters,
            getattr(config, "utility_beta", config.beta),
            config.objective,
            config.quality_threshold,
            prefer_transformer=config.prefer_transformer, embedding_dim=config.embedding_dim,
        )

        # todo 是不是有点问题这块
        self.fuser_torch: Optional[MLPFuserTorch] = None
        # 训练期融合输入的归一化参数，推理/评估阶段复用
        self._norm_min: np.ndarray | None = None
        self._norm_max: np.ndarray | None = None
        self._query_feature_extractor: QueryFeatureExtractor | None = None
        if getattr(config, "use_query_features", True):
            self._query_feature_extractor = QueryFeatureExtractor(
                hash_buckets=getattr(config, "query_hash_buckets", 128),
                max_hash_per_ngram=getattr(config, "query_max_hash_per_ngram", 64),
                spacy_model=getattr(config, "spacy_model", "en_core_web_sm"),
                spacy_components=getattr(config, "spacy_enable_components", ("tok2vec", "tagger", "parser", "ner")),
            )
        self._embed_cluster_model: object | None = None
        self._embed_cluster_key: str | None = None
        self._limbo_summary: Dict[str, Any] | None = None
        self._limbo_cluster_profiles: List[Dict[str, object]] | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self._norm_min is not None and self._norm_max is not None:
            denom = self._norm_max - self._norm_min
            denom = np.where(denom == 0, 1.0, denom)
            return (x - self._norm_min) / denom
        if x.shape[0] <= 1:
            return x
        min_v = x.min(axis=0, keepdims=True)
        max_v = x.max(axis=0, keepdims=True)
        denom = (max_v - min_v)
        denom[denom == 0] = 1.0
        return (x - min_v) / denom

    def _extract_query_features(self, texts: Sequence[str]) -> List[Dict[str, str]]:
        if self._query_feature_extractor is None:
            return [dict() for _ in texts]
        return self._query_feature_extractor.extract(list(texts))

    def _merge_features(
        self,
        original: Sequence[Optional[Dict[str, str]]],
        auto_features: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        for base, extra in zip(original, auto_features):
            combined: Dict[str, str] = {}
            if base:
                for k, v in base.items():
                    combined[str(k)] = "" if v is None else str(v)
            combined.update(extra)
            merged.append(combined)
        return merged

    def _fit_embed_clusters(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        if embeddings.size == 0 or embeddings.shape[0] == 0:
            self._embed_cluster_model = None
            self._embed_cluster_key = None
            return np.zeros((embeddings.shape[0],), dtype=np.int32)
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("启用 query_embed_k 需要安装 scikit-learn：pip install scikit-learn") from exc
        random_state = getattr(self.config, "seed", 42)
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings.astype(np.float32))
        self._embed_cluster_model = kmeans
        self._embed_cluster_key = f"embed_cluster_{k}"
        return labels.astype(np.int32)

    def _predict_embed_clusters(self, texts: Sequence[str]) -> np.ndarray:
        if self._embed_cluster_model is None or self._embed_cluster_key is None:
            raise RuntimeError("嵌入簇模型尚未训练，无法推断 embed_cluster 特征")
        embeddings = self.bert_branch.transform_texts(list(texts), show_progress_bar=False)
        if embeddings.size == 0:
            return np.zeros((len(texts),), dtype=np.int32)
        model = self._embed_cluster_model  # type: ignore[assignment]
        labels = model.predict(embeddings)  # type: ignore[attr-defined]
        return np.asarray(labels, dtype=np.int32)

    def _attach_embed_labels(self, auto_features: Sequence[Dict[str, str]], labels: np.ndarray) -> None:
        if self._embed_cluster_key is None:
            return
        for feat, label in zip(auto_features, labels):
            feat[self._embed_cluster_key] = str(int(label))

    def fit(self, queries_text: List[str], queries_features: List[Dict[str, str]], quality: Optional[np.ndarray] = None) -> None:
        n = len(queries_text)
        m = len(self.model_names)
        try:
            print(f"[RouterPipeline.fit] start: n_texts={n}, n_models={m}", flush=True)
            print(f"[RouterPipeline.fit] config: fuser_type={self.config.fuser_type}, num_clusters={self.config.num_clusters}, fusion_beta={getattr(self.config, 'fusion_beta', None)}", flush=True)
        except Exception:
            pass
        if quality is None:
            quality = np.random.rand(n, m).astype(np.float32)
        cost = np.asarray(self.model_costs, dtype=np.float32)

        show_progress = bool(getattr(self.config, "show_progress", True))

        auto_features = self._extract_query_features(queries_text)
        self._embed_cluster_model = None
        self._embed_cluster_key = None

        if show_progress:
            print("[RouterPipeline.fit] BERT branch: encoding & fitting ...", flush=True)
        bert_res = self.bert_branch.fit(queries_text, quality)
        try:
            print(f"[RouterPipeline.fit] BERT done: rep_shape={getattr(bert_res, 'representation', np.empty((0, 0))).shape}", flush=True)
        except Exception:
            pass

        embed_k = getattr(self.config, "query_embed_k", None)
        if embed_k is not None:
            labels = self._fit_embed_clusters(bert_res.representation, int(embed_k))
            self._attach_embed_labels(auto_features, labels)

        merged_features = self._merge_features(queries_features, auto_features)
        if show_progress:
            print(f"[RouterPipeline.fit] merged LIMBO feature example: keys={list(merged_features[0].keys())[:10]}", flush=True)

        if show_progress:
            print("[RouterPipeline.fit] LIMBO branch: fitting ...", flush=True)
        limbo_res = self.limbo_branch.fit(merged_features, quality)
        try:
            print(f"[RouterPipeline.fit] LIMBO done: rep_shape={getattr(limbo_res, 'representation', np.empty((0, 0))).shape}", flush=True)
        except Exception:
            pass
        try:
            summary = self.limbo_branch.summary(top_k=5)
            self._limbo_summary = summary
            print(f"[RouterPipeline.fit] LIMBO summary: {summary}", flush=True)
        except Exception:
            self._limbo_summary = None
        try:
            profiles = self.limbo_branch.cluster_profiles()
            self._limbo_cluster_profiles = profiles
            preview = profiles[: min(3, len(profiles))]
            print(f"[RouterPipeline.fit] LIMBO cluster profiles preview: {preview}", flush=True)
        except Exception:
            self._limbo_cluster_profiles = None

        # 2) 生成软标签（分支）
        y_limbo = limbo_res.soft_labels
        y_bert = bert_res.soft_labels
        fusion_beta = float(getattr(self.config, "fusion_beta", self.config.lambda_soft))
        fusion_beta = max(0.0, min(1.0, fusion_beta))
        y_branch = fusion_beta * y_limbo + (1.0 - fusion_beta) * y_bert
        try:
            print(f"[RouterPipeline.fit] branch soft labels prepared: shape={y_branch.shape}, fusion_beta={fusion_beta}", flush=True)
        except Exception:
            pass

        # 2.1) 成本感知目标分布：使用质量(0/1)与成本
        alpha = float(getattr(self.config, "cost_weight", 0.5))
        tau = float(getattr(self.config, "softmax_temperature", 0.5))
        gamma = float(getattr(self.config, "blend_costaware", 0.5))
        # 归一化成本
        cost_norm = cost / max(1e-12, float(cost.max()))
        # logits = (quality - alpha * cost_norm) / tau
        q = quality.astype(np.float32)
        logits = (q - cost_norm.reshape(1, -1) * alpha) / max(1e-6, tau)
        logits = logits - logits.max(axis=1, keepdims=True)
        p_exp = np.exp(logits)
        denom = p_exp.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        p_costaware = p_exp / denom

        # 2.2) 融合目标
        y_target = gamma * p_costaware + (1.0 - gamma) * y_branch
        try:
            print(f"[RouterPipeline.fit] cost-aware target prepared: shape={y_target.shape} (alpha={alpha}, tau={tau}, gamma={gamma})", flush=True)
        except Exception:
            pass

        # 3) 融合输入
        X_fuse = np.hstack([limbo_res.representation, bert_res.representation])
        # 门控缩放：先缩放后归一化统计
        limbo_dim = limbo_res.representation.shape[1]
        bert_dim = bert_res.representation.shape[1]
        if X_fuse.size > 0 and (limbo_dim + bert_dim) == X_fuse.shape[1]:
            X_fuse = X_fuse.copy()
            X_fuse[:, :limbo_dim] *= fusion_beta
            X_fuse[:, limbo_dim:] *= (1.0 - fusion_beta)
        # 记录训练期 min/max，用于推理/评估阶段归一化保持一致
        if X_fuse.size > 0:
            self._norm_min = X_fuse.min(axis=0, keepdims=True)
            self._norm_max = X_fuse.max(axis=0, keepdims=True)
        try:
            print(f"[RouterPipeline.fit] fusion features ready: shape={X_fuse.shape}", flush=True)
        except Exception:
            pass

        # 4) 融合
        import torch
        if self.config.fuser_type == "attention":
            limbo_dim = limbo_res.representation.shape[1]
            bert_dim = bert_res.representation.shape[1]
            self.fuser_torch = AttentionFuserTorch(limbo_dim, bert_dim, num_models=m)
            x_t = torch.from_numpy(X_fuse.astype(np.float32))
            y_t = torch.from_numpy(y_target.astype(np.float32))
            # 共享训练函数
            print(f"[RouterPipeline.fit] training Attention fuser: samples={x_t.size(0)}, epochs={self.config.fuser_epochs}, lr={self.config.fuser_lr}", flush=True)
            train_fuser(self.fuser_torch, x_t, y_t, epochs=self.config.fuser_epochs, lr=self.config.fuser_lr, use_tqdm=getattr(self.config, "show_progress", True))
        else:
            self.fuser_torch = MLPFuserTorch(input_dim=X_fuse.shape[1], num_models=m, hidden_dims=self.config.fuser_hidden)
            print(f"[RouterPipeline.fit] training MLP fuser: samples={X_fuse.shape[0]}, input_dim={X_fuse.shape[1]}, epochs={self.config.fuser_epochs}, lr={self.config.fuser_lr}", flush=True)
            train_fuser(
                self.fuser_torch,
                torch.from_numpy(X_fuse.astype(np.float32)),
                torch.from_numpy(y_target.astype(np.float32)),
                epochs=self.config.fuser_epochs,
                lr=self.config.fuser_lr,
                use_tqdm=getattr(self.config, "show_progress", True),
            )
        try:
            print("[RouterPipeline.fit] fuser training finished", flush=True)
        except Exception:
            pass

    def _fit_mapper(self, mapper: ClusterModelMapper, labels: List[int], quality: np.ndarray) -> None:
        # temp temp
        if self.config.objective == "min_cost":
            mapper.fit_min_cost(labels, quality, self.config.quality_threshold)
        else:
            mapper.fit(labels, quality, self.config.beta)

    @property
    def limbo_summary(self) -> Optional[Dict[str, Any]]:
        return self._limbo_summary

    @property
    def limbo_cluster_profiles(self) -> Optional[List[Dict[str, float]]]:
        return self._limbo_cluster_profiles

    def predict(self, query_text: str, query_features: Dict[str, str]) -> Tuple[str, float, int]:
        if self.fuser_torch is None:
            raise RuntimeError("Pipeline not fitted")

        auto_feats = self._extract_query_features([query_text])
        if self._embed_cluster_key is not None:
            labels = self._predict_embed_clusters([query_text])
            self._attach_embed_labels(auto_feats, labels)
        merged = self._merge_features([query_features], auto_feats)

        # fusion here：先拼接，再用训练期统计量统一归一化
        x_limbo = self.limbo_branch.vectorizer.transform(merged)
        x_bert = self.bert_branch.transform_texts([query_text], show_progress_bar=False)
        x = np.hstack([x_limbo, x_bert])
        # 门控缩放：与训练一致
        fusion_beta = float(getattr(self.config, "fusion_beta", self.config.lambda_soft))
        fusion_beta = max(0.0, min(1.0, fusion_beta))
        limbo_dim = x_limbo.shape[1]
        bert_dim = x_bert.shape[1]
        if x.size > 0 and (limbo_dim + bert_dim) == x.shape[1]:
            x = x.copy()
            x[:, :limbo_dim] *= fusion_beta
            x[:, limbo_dim:] *= (1.0 - fusion_beta)
        x = self._normalize(x)

        import torch

        with torch.no_grad():
            logp = self.fuser_torch(torch.from_numpy(x.astype(np.float32)))
            probs = logp.exp().numpy()[0]
        idx = int(np.argmax(probs))
        return self.model_names[idx], float(probs[idx]), idx

    # evaluate
    def evaluate(self, texts: List[str], features: List[Dict[str, str]], labels: Optional[List[int]] = None, quality: Optional[List[List[float]]] = None, sample_costs: Optional[List[List[float]]] = None) -> Dict[str, float]:
        try:
            print(f"[RouterPipeline.evaluate] start: n={len(texts)}", flush=True)
        except Exception:
            pass
        hits = 0
        costs: List[float] = []
        preds: List[int] = []
        probs_list: List[np.ndarray] = []
        route_hist = {name: 0 for name in self.model_names}

        auto_feats = self._extract_query_features(texts)
        if self._embed_cluster_key is not None:
            labels_embed = self._predict_embed_clusters(texts)
            self._attach_embed_labels(auto_feats, labels_embed)
        merged_features = self._merge_features(features, auto_feats)

        x_limbo = self.limbo_branch.vectorizer.transform(merged_features)
        x_bert = self.bert_branch.transform_texts(texts, show_progress_bar=False)
        xn = np.hstack([x_limbo, x_bert])
        try:
            print(f"[RouterPipeline.evaluate] features ready: limbo={x_limbo.shape}, bert={x_bert.shape}, fused={xn.shape}", flush=True)
        except Exception:
            pass
        fusion_beta = float(getattr(self.config, "fusion_beta", self.config.lambda_soft))
        fusion_beta = max(0.0, min(1.0, fusion_beta))
        limbo_dim = x_limbo.shape[1]
        bert_dim = x_bert.shape[1]
        if xn.size > 0 and (limbo_dim + bert_dim) == xn.shape[1]:
            xn = xn.copy()
            xn[:, :limbo_dim] *= fusion_beta
            xn[:, limbo_dim:] *= (1.0 - fusion_beta)
        xn = self._normalize(xn)

        import torch
        iterator = range(len(texts))
        if getattr(self.config, "show_progress", True):
            iterator = tqdm(iterator, desc="Evaluating", leave=False)
        with torch.no_grad():
            for i in iterator:
                logp = self.fuser_torch(torch.from_numpy(xn[i:i+1].astype(np.float32)))  # type: ignore[arg-type]
                probs = logp.exp().numpy()[0]
                probs_list.append(probs)
                idx = int(np.argmax(probs))
                preds.append(idx)
                if sample_costs is not None and i < len(sample_costs):
                    row = sample_costs[i]
                    if isinstance(row, (list, tuple)) and len(row) == len(self.model_names):
                        costs.append(float(row[idx]))
                    else:
                        costs.append(float(self.model_costs[idx]))
                else:
                    costs.append(float(self.model_costs[idx]))
                route_hist[self.model_names[idx]] += 1
                if labels is not None and i < len(labels):
                    hits += int(idx == labels[i])
        
        metrics: Dict[str, float] = {"avg_cost": float(np.mean(costs))}
        total = max(1, len(texts))
        for name, cnt in route_hist.items():
            metrics[f"route_{name}"] = cnt / total

        # Quality Metrics
        if quality is not None and len(quality) == len(preds):
            q_arr = np.asarray(quality, dtype=np.float32)
            # Avg Quality
            pred_q = q_arr[np.arange(len(preds)), preds]
            avg_q = float(pred_q.mean())
            metrics["avg_quality"] = avg_q
            
            # Oracle Quality
            oracle_q = q_arr.max(axis=1).mean()
            metrics["oracle_quality"] = float(oracle_q)
            
            # Regret & Retention
            metrics["quality_regret"] = float(oracle_q - avg_q)
            metrics["quality_retention"] = float(avg_q / oracle_q) if oracle_q > 1e-9 else 1.0

        # Cost Metrics
        if self.model_costs:
            max_cost = max(self.model_costs)
            metrics["cost_savings"] = float((max_cost - metrics["avg_cost"]) / max_cost) if max_cost > 1e-9 else 0.0

        # Classification Metrics (requires labels)
        if labels is not None and len(labels) == len(preds) and len(labels) > 0:
            labels_np = np.asarray(labels, dtype=np.int64)
            preds_np = np.asarray(preds, dtype=np.int64)
            
            # Top-1 Accuracy
            top1_acc = float((preds_np == labels_np).mean())
            metrics["top1_acc"] = top1_acc
            
            # Cost per Correct
            metrics["cost_per_correct"] = metrics["avg_cost"] / max(1e-9, top1_acc)

            # Top-k Accuracy
            if probs_list:
                probs_np = np.asarray(probs_list)
                for k in [3, 5]:
                    if k <= len(self.model_names):
                        top_k_idx = np.argsort(probs_np, axis=1)[:, -k:]
                        top_k_hits = np.any(top_k_idx == labels_np[:, None], axis=1).mean()
                        metrics[f"top{k}_acc"] = float(top_k_hits)
            
            # Advanced Classification: F1 & MCC
            # Simple numpy implementation to avoid heavy sklearn dependency if not needed, 
            # but for MCC/F1 sklearn is robust.
            try:
                from sklearn.metrics import f1_score, matthews_corrcoef # type: ignore
                metrics["macro_f1"] = float(f1_score(labels_np, preds_np, average="macro"))
                metrics["weighted_f1"] = float(f1_score(labels_np, preds_np, average="weighted"))
                metrics["mcc"] = float(matthews_corrcoef(labels_np, preds_np))
            except ImportError:
                pass
            except Exception:
                pass

            # Distribution Metrics: KL Divergence
            # P: predicted distribution (histogram), Q: label distribution (histogram)
            # Add smoothing to avoid log(0)
            n_classes = len(self.model_names)
            p_hist = np.bincount(preds_np, minlength=n_classes).astype(np.float32) / total
            q_hist = np.bincount(labels_np, minlength=n_classes).astype(np.float32) / total
            epsilon = 1e-9
            p_hist = np.clip(p_hist, epsilon, 1.0)
            q_hist = np.clip(q_hist, epsilon, 1.0)
            # Normalize again
            p_hist /= p_hist.sum()
            q_hist /= q_hist.sum()
            kl = np.sum(p_hist * np.log(p_hist / q_hist))
            metrics["kl_div"] = float(kl)

            # --- Existing Baselines Logic ---
            # 固定路由至每个模型的基线：准确率与单位成本
            fixed_accs: List[float] = []
            for model_idx in range(len(self.model_names)):
                acc_i = float((labels_np == model_idx).mean())
                fixed_accs.append(acc_i)
                metrics[f"fixed_acc_{self.model_names[model_idx]}"] = acc_i
                metrics[f"fixed_cost_{self.model_names[model_idx]}"] = float(self.model_costs[model_idx])

            # 最佳固定模型（按准确率）
            if fixed_accs:
                best_fixed_idx = int(np.argmax(np.asarray(fixed_accs)))
                metrics["baseline_best_fixed_model"] = float(best_fixed_idx)
                metrics["baseline_best_fixed_model_name"] = self.model_names[best_fixed_idx]
                metrics["baseline_best_fixed_acc"] = fixed_accs[best_fixed_idx]
                metrics["baseline_best_fixed_cost"] = float(self.model_costs[best_fixed_idx])

            # 匹配当前准确率的最低成本固定模型（若存在）
            eps = 1e-12
            candidates = [
                (i, float(self.model_costs[i]))
                for i, acc in enumerate(fixed_accs)
                if acc + eps >= top1_acc
            ]
            if candidates:
                matched_idx, matched_cost = min(candidates, key=lambda x: x[1])
                metrics["matched_perf_min_cost_model"] = float(matched_idx)
                metrics["matched_perf_min_cost_model_name"] = self.model_names[matched_idx]
                metrics["matched_perf_min_cost"] = matched_cost
                metrics["cost_ratio_vs_matched"] = metrics["avg_cost"] / max(matched_cost, eps)
                metrics["meets_cost_minimization_under_perf"] = float(metrics["avg_cost"] <= matched_cost)
            else:
                metrics["matched_perf_min_cost_model"] = -1.0
                metrics["matched_perf_min_cost_model_name"] = "none"
                metrics["matched_perf_min_cost"] = float("inf")
                metrics["cost_ratio_vs_matched"] = 0.0
                metrics["meets_cost_minimization_under_perf"] = 1.0

            # 最便宜固定模型（参考）
            if len(self.model_costs) > 0:
                if sample_costs is not None and len(sample_costs) == len(texts):
                    costs_mat = np.asarray(sample_costs, dtype=np.float32)
                    per_model_avg = costs_mat.mean(axis=0)
                    min_cost_idx = int(np.argmin(per_model_avg))
                    metrics["baseline_cheapest_model"] = float(min_cost_idx)
                    metrics["baseline_cheapest_model_name"] = self.model_names[min_cost_idx]
                    metrics["baseline_cheapest_cost"] = float(per_model_avg[min_cost_idx])
                else:
                    costs_arr = np.asarray(self.model_costs, dtype=np.float32)
                    min_cost_idx = int(np.argmin(costs_arr))
                    metrics["baseline_cheapest_model"] = float(min_cost_idx)
                    metrics["baseline_cheapest_model_name"] = self.model_names[min_cost_idx]
                    metrics["baseline_cheapest_cost"] = float(costs_arr[min_cost_idx])

        return metrics

    # save and load
    def save(self, path: str) -> None:
        fuser_state = None
        if self.fuser_torch is not None:
            fuser_state = {k: v.detach().cpu().numpy().tolist() for k, v in self.fuser_torch.state_dict().items()}
        data = {
            "config": self.config.__dict__,
            "model_names": self.model_names,
            "model_costs": self.model_costs,
            "limbo_mapping": self.limbo_branch.mapper.mapping,
            "bert_mapping": self.bert_branch.mapper.mapping,
            "fuser_torch": fuser_state,
        }
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load(path: str) -> "RouterPipeline":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = TrainConfig(**data["config"])  # type: ignore[arg-type]
        pipe = RouterPipeline(data["model_names"], data["model_costs"], cfg)
        pipe.limbo_branch.mapper.mapping = {int(k): v for k, v in data["limbo_mapping"].items()}
        pipe.bert_branch.mapper.mapping = {int(k): v for k, v in data["bert_mapping"].items()}
        state = data.get("fuser_torch")
        if state is not None:
            import torch

            # 需要先构建形状，简单起见用隐藏维度默认
            # 不冲训练！！！
            # 由于输入维度未知，无法在无数据时构造；在第一次fit后再保存。
            # 实际shape由state_dict覆盖。
            # 从 state 中推断输入与输出维度
            w0 = np.asarray(state.get("fc1.weight"))
            in_dim = int(w0.shape[1]) if w0 is not None else 1
            out_dim = len(pipe.model_names)
            pipe.fuser_torch = MLPFuserTorch(input_dim=in_dim, num_models=out_dim, hidden_dims=pipe.config.fuser_hidden)
            state_t = {k: torch.tensor(v) for k, v in state.items()}
            pipe.fuser_torch.load_state_dict(state_t)
        return pipe

