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


@dataclass
class TrainConfig(TrainConfig):
    pass


class RouterPipeline:
    """
    端到端路由流水线：
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
        limbo_res = self.limbo_branch.fit(queries_features, quality)
        bert_res = self.bert_branch.fit(queries_text, quality)

        # 2) 生成软标签
        y_limbo = limbo_res.soft_labels
        y_bert = bert_res.soft_labels
        y_target = self.config.lambda_soft * y_limbo + (1.0 - self.config.lambda_soft) * y_bert

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
            train_fuser(self.fuser_torch, x_t, y_t, epochs=self.config.fuser_epochs, lr=self.config.fuser_lr)
        else:
            self.fuser_torch = MLPFuserTorch(input_dim=X_fuse.shape[1], num_models=m, hidden_dims=self.config.fuser_hidden)
            train_fuser(
                self.fuser_torch,
                torch.from_numpy(X_fuse.astype(np.float32)),
                torch.from_numpy(y_target.astype(np.float32)),
                epochs=self.config.fuser_epochs,
                lr=self.config.fuser_lr,
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

        # fusion here
        x_limbo = self.limbo_branch.vectorizer.transform([query_features])
        x_bert = self.bert_branch.encoder.encode([query_text])
        x = np.hstack([self._normalize(x_limbo), self._normalize(x_bert)])

        import torch

        with torch.no_grad():
            logp = self.fuser_torch(torch.from_numpy(x.astype(np.float32)))
            probs = logp.exp().numpy()[0]
        idx = int(np.argmax(probs))
        return self.model_names[idx], float(probs[idx]), idx

    # evaluate
    def evaluate(self, texts: List[str], features: List[Dict[str, str]], labels: Optional[List[int]] = None) -> Dict[str, float]:
        hits = 0
        costs = []
        route_hist = {name: 0 for name in self.model_names}
        for i in range(len(texts)):
            _, _, idx = self.predict(texts[i], features[i])
            costs.append(float(self.model_costs[idx]))
            route_hist[self.model_names[idx]] += 1
            if labels is not None and i < len(labels):
                hits += int(idx == labels[i])
        metrics = {"avg_cost": float(np.mean(costs))}
        if labels is not None:
            metrics["top1_acc"] = hits / max(1, len(labels))
        total = max(1, len(texts))
        for name, cnt in route_hist.items():
            metrics[f"route_{name}"] = cnt / total
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

