from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .encoders import CategoricalFeatureVectorizer, build_text_encoder
from .mapping import ClusterModelMapper
from .branches.limbo_branch import LimboBranch
from .branches.bert_branch import BertBranch
from .models.fusion_torch import MLPFuserTorch, AttentionFuserTorch, train_fuser
from limbo_cluster import LimboAgglomerative  
from .logger import RunLogger, RunInfo
from .core import TrainConfig
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
# 也许我们可以把所有tunable的hyperparameter都写在这边？
# 目前有你的那三个可以控制weighting of cost and performance
# 现在加了beta
# 还差LIMBO的tau。然后再做grid search？
class TrainConfig(TrainConfig):
    beta: float = 0.5 # alpha controls the weighting. 1 means all LIMBO, 0 means all BERT.


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
        self.limbo_branch = LimboBranch(
            model_names, model_costs, config.num_clusters, config.beta, config.objective, config.quality_threshold
        )
        self.bert_branch = BertBranch(
            model_names, model_costs, config.num_clusters, config.beta, config.objective, config.quality_threshold,
            prefer_transformer=config.prefer_transformer, embedding_dim=config.embedding_dim,
        )

        # todo 是不是有点问题这块
        self.fuser_torch: Optional[MLPFuserTorch] = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        min_v = x.min(axis=0, keepdims=True)
        max_v = x.max(axis=0, keepdims=True)
        denom = (max_v - min_v)
        denom[denom == 0] = 1.0
        return (x - min_v) / denom

    def fit(self, queries_text: List[str], queries_features: List[Dict[str, str]], quality: Optional[np.ndarray] = None) -> None:
        n = len(queries_text)
        m = len(self.model_names)
        if quality is None:
            quality = np.random.rand(n, m).astype(np.float32)
        cost = np.asarray(self.model_costs, dtype=np.float32)

        # 1) 两个分支
        if getattr(self.config, "show_progress", True):
            limbo_res = self.limbo_branch.fit(queries_features, quality)
            bert_res = self.bert_branch.fit(queries_text, quality)
        else:
            limbo_res = self.limbo_branch.fit(queries_features, quality)
            bert_res = self.bert_branch.fit(queries_text, quality)

        # 2) 生成软标签（分支）
        y_limbo = limbo_res.soft_labels
        y_bert = bert_res.soft_labels
        y_branch = self.config.lambda_soft * y_limbo + (1.0 - self.config.lambda_soft) * y_bert

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

        # 3) 融合输入
        X_fuse = np.hstack([limbo_res.representation, bert_res.representation])

        # 4) 融合
        import torch
        if self.config.fuser_type == "attention":
            limbo_dim = limbo_res.representation.shape[1]
            bert_dim = bert_res.representation.shape[1]
            self.fuser_torch = AttentionFuserTorch(limbo_dim, bert_dim, num_models=m)
            x_t = torch.from_numpy(X_fuse.astype(np.float32))
            y_t = torch.from_numpy(y_target.astype(np.float32))
            # 共享训练函数
            train_fuser(self.fuser_torch, x_t, y_t, epochs=self.config.fuser_epochs, lr=self.config.fuser_lr, use_tqdm=getattr(self.config, "show_progress", True))
        else:
            self.fuser_torch = MLPFuserTorch(input_dim=X_fuse.shape[1], num_models=m, hidden_dims=self.config.fuser_hidden)
            train_fuser(
                self.fuser_torch,
                torch.from_numpy(X_fuse.astype(np.float32)),
                torch.from_numpy(y_target.astype(np.float32)),
                epochs=self.config.fuser_epochs,
                lr=self.config.fuser_lr,
                use_tqdm=getattr(self.config, "show_progress", True),
            )

    def _fit_mapper(self, mapper: ClusterModelMapper, labels: List[int], quality: np.ndarray) -> None:
        # temp temp
        if self.config.objective == "min_cost":
            mapper.fit_min_cost(labels, quality, self.config.quality_threshold)
        else:
            mapper.fit(labels, quality, self.config.beta)

    def predict(self, query_text: str, query_features: Dict[str, str]) -> Tuple[str, float, int]:
        if self.fuser_torch is None:
            raise RuntimeError("Pipeline not fitted")
        
        # fetching beta
        beta = float(getattr(self.config, "beta", 0.5))
        if beta < 0 or beta > 1:
            raise RuntimeError("Beta must be in the range of [0, 1].")

        # fusion here
        x_limbo = self.limbo_branch.vectorizer.transform([query_features])
        # 单条预测禁用编码器内部进度，避免不连贯
        x_bert = self.bert_branch.encoder.encode([query_text], show_progress_bar=False)
        # fusing two branches with beta
        x = np.hstack([
            self._normalize(x_limbo) * beta, 
            self._normalize(x_bert) * (1 - beta),
        ])

        import torch

        with torch.no_grad():
            logp = self.fuser_torch(torch.from_numpy(x.astype(np.float32)))
            probs = logp.exp().numpy()[0]
        idx = int(np.argmax(probs))
        return self.model_names[idx], float(probs[idx]), idx

    # evaluate
    def evaluate(self, texts: List[str], features: List[Dict[str, str]], labels: Optional[List[int]] = None) -> Dict[str, float]:
        hits = 0
        costs: List[float] = []
        preds: List[int] = []
        route_hist = {name: 0 for name in self.model_names}

        x_limbo = self.limbo_branch.vectorizer.transform(features)
        x_bert = self.bert_branch.encoder.encode(texts, show_progress_bar=False)
        xn = np.hstack([self._normalize(x_limbo), self._normalize(x_bert)])

        import torch
        iterator = range(len(texts))
        if getattr(self.config, "show_progress", True):
            iterator = tqdm(iterator, desc="Evaluating", leave=False)
        with torch.no_grad():
            for i in iterator:
                logp = self.fuser_torch(torch.from_numpy(xn[i:i+1].astype(np.float32)))  # type: ignore[arg-type]
                probs = logp.exp().numpy()[0]
                idx = int(np.argmax(probs))
                preds.append(idx)
                costs.append(float(self.model_costs[idx]))
                route_hist[self.model_names[idx]] += 1
                if labels is not None and i < len(labels):
                    hits += int(idx == labels[i])
        metrics: Dict[str, float] = {"avg_cost": float(np.mean(costs))}
        total = max(1, len(texts))
        for name, cnt in route_hist.items():
            metrics[f"route_{name}"] = cnt / total

        # 若提供标签，计算扩展指标用于“在保持性能的前提下最小化成本”的可行性分析
        if labels is not None and len(labels) == len(preds) and len(labels) > 0:
            labels_np = np.asarray(labels, dtype=np.int64)
            preds_np = np.asarray(preds, dtype=np.int64)
            top1_acc = float((preds_np == labels_np).mean())
            metrics["top1_acc"] = top1_acc

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

