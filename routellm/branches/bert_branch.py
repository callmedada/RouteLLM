from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..encoders import build_text_encoder
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x
from ..mapping import ClusterModelMapper
from limbo_cluster import LimboAgglomerative


@dataclass
class BranchResult:
    labels: List[int]
    representation: np.ndarray
    soft_labels: np.ndarray


class BertBranch:
    def __init__(self, model_names: List[str], model_costs: List[float], num_clusters: int, beta: float, objective: str = "utility", quality_threshold: float = 0.0, prefer_transformer: bool = True, embedding_dim: int = 384) -> None:
        self.model_names = model_names
        self.model_costs = model_costs
        self.num_clusters = num_clusters
        self.beta = beta
        self.objective = objective
        self.quality_threshold = quality_threshold

        self.encoder = build_text_encoder(prefer_transformer=prefer_transformer, embedding_dim=embedding_dim)
        self.clusterer = LimboAgglomerative(n_clusters=num_clusters)
        self.mapper = ClusterModelMapper(model_names, model_costs)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        min_v = x.min(axis=0, keepdims=True)
        max_v = x.max(axis=0, keepdims=True)
        denom = max_v - min_v
        denom[denom == 0] = 1.0
        return (x - min_v) / denom

    def fit(self, texts: List[str], quality: np.ndarray) -> BranchResult:
        # 大批量时让底层 encoder 显示进度，小批量不显示，避免条目不连贯
        X = self.encoder.encode(texts, show_progress_bar=len(texts) >= 32)
        dicts = []
        for vec in tqdm(X, desc="BERT branch normalize", leave=False):
            v = np.asarray(vec, dtype=np.float32)
            v = v - v.min()  # 移动到非负
            s = float(v.sum())
            if s <= 0:
                v = np.ones_like(v) / max(1, len(v))
            else:
                v = v / s
            d = {f"f{j}": float(vj) for j, vj in enumerate(v)}
            dicts.append(d)
        self.clusterer.fit(dicts)
        labels = list(self.clusterer.labels_)
        Xn = self._normalize(X)

        if self.objective == "min_cost":
            self.mapper.fit_min_cost(labels, quality, self.quality_threshold)
        else:
            self.mapper.fit(labels, quality, self.beta)

        y = np.zeros((len(labels), len(self.model_names)), dtype=np.float32)
        for i, c in enumerate(labels):
            if self.objective == "min_cost":
                q = quality[i]
                ok = q >= self.quality_threshold
                if ok.any():
                    inv_cost = np.where(ok, 1.0 / np.asarray(self.model_costs, dtype=np.float32), 0.0)
                    s = inv_cost.sum()
                    if s > 0:
                        y[i] = inv_cost / s
                        continue
            name = self.mapper.mapping[c]
            y[i, self.model_names.index(name)] = 1.0

        return BranchResult(labels=labels, representation=Xn, soft_labels=y)

