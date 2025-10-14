from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from limbo_cluster import LimboAgglomerative

from ..encoders import CategoricalFeatureVectorizer
from ..mapping import ClusterModelMapper
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
class BranchResult:
    labels: List[int]
    representation: np.ndarray
    soft_labels: np.ndarray


class LimboBranch:
    """
    LIMBO
    """

    def __init__(self, model_names: List[str], model_costs: List[float], num_clusters: int, beta: float, objective: str = "utility", quality_threshold: float = 0.0, *, limbo_tau: float | None = None, limbo_use_sparse: bool = True) -> None:
        self.model_names = model_names
        self.model_costs = model_costs
        self.num_clusters = num_clusters
        self.beta = beta
        self.objective = objective
        self.quality_threshold = quality_threshold

        self.clusterer = LimboAgglomerative(n_clusters=num_clusters, tau=limbo_tau, use_sparse=limbo_use_sparse)
        self.vectorizer = CategoricalFeatureVectorizer()
        self.mapper = ClusterModelMapper(model_names, model_costs)
        self.norm_min: np.ndarray | None = None
        self.norm_max: np.ndarray | None = None

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        min_v = x.min(axis=0, keepdims=True)
        max_v = x.max(axis=0, keepdims=True)
        denom = max_v - min_v
        denom[denom == 0] = 1.0
        return (x - min_v) / denom

    def fit(self, features: List[Dict[str, str]], quality: np.ndarray) -> BranchResult:
        # 聚类
        try:
            print(f"[LimboBranch.fit] start: n={len(features)}, num_clusters={self.num_clusters}", flush=True)
        except Exception:
            pass
        self.clusterer.fit(features)
        labels = list(self.clusterer.labels_)

        print("[LimboBranch.fit] vectorizing features ...", flush=True)
        self.vectorizer.fit(features)
        X_raw = self.vectorizer.transform(features)
        if X_raw.size == 0:
            X = X_raw
        else:
            min_v = X_raw.min(axis=0, keepdims=True)
            max_v = X_raw.max(axis=0, keepdims=True)
            denom = max_v - min_v
            denom[denom == 0] = 1.0
            self.norm_min = min_v
            self.norm_max = max_v
            X = (X_raw - min_v) / denom
        print(f"[LimboBranch.fit] normed feature shape={X.shape}", flush=True)
        if self.objective == "min_cost":
            self.mapper.fit_min_cost(labels, quality, self.quality_threshold)
        else:
            # 使用 utility_beta（若未提供则等于 legacy beta）
            beta = getattr(self, "beta", 0.0)
            self.mapper.fit(labels, quality, beta)

        #  1/cost normalize
        y = np.zeros((len(labels), len(self.model_names)), dtype=np.float32)
        for i, c in enumerate(tqdm(labels, desc="LIMBO branch labels", leave=False)):
            if self.objective == "min_cost":
                q = quality[i]
                ok = q >= self.quality_threshold
                if ok.any():
                    inv_cost = np.where(ok, 1.0 / np.asarray(self.model_costs, dtype=np.float32), 0.0)
                    s = inv_cost.sum()
                    if s > 0:
                        y[i] = inv_cost / s
                        continue
            # back tp one-hot
            name = self.mapper.mapping[c]
            y[i, self.model_names.index(name)] = 1.0
        print(f"[LimboBranch.fit] done: labels={len(labels)}, y_shape={y.shape}", flush=True)
        return BranchResult(labels=labels, representation=X, soft_labels=y)

