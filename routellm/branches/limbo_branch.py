from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from limbo_cluster import LimboAgglomerative

from ..encoders import CategoricalFeatureVectorizer
from ..mapping import ClusterModelMapper


@dataclass
class BranchResult:
    labels: List[int]
    representation: np.ndarray
    soft_labels: np.ndarray


class LimboBranch:
    """
    LIMBO 分支：
    - 使用外部 LimboAgglomerative 对原始离散特征字典聚类
    - 使用 CategoricalFeatureVectorizer 生成稠密 one-hot 表征用于融合
    - 使用 ClusterModelMapper 完成簇到模型的映射
    """

    def __init__(self, model_names: List[str], model_costs: List[float], num_clusters: int, beta: float, objective: str = "utility", quality_threshold: float = 0.0) -> None:
        self.model_names = model_names
        self.model_costs = model_costs
        self.num_clusters = num_clusters
        self.beta = beta
        self.objective = objective
        self.quality_threshold = quality_threshold

        self.clusterer = LimboAgglomerative(n_clusters=num_clusters)
        self.vectorizer = CategoricalFeatureVectorizer()
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

    def fit(self, features: List[Dict[str, str]], quality: np.ndarray) -> BranchResult:
        # 聚类（基于原始字典）
        self.clusterer.fit(features)
        labels = list(self.clusterer.labels_)

        # 特征表征用于融合
        self.vectorizer.fit(features)
        X = self._normalize(self.vectorizer.transform(features))

        # 簇-模型映射
        if self.objective == "min_cost":
            self.mapper.fit_min_cost(labels, quality, self.quality_threshold)
        else:
            self.mapper.fit(labels, quality, self.beta)

        # 成本感知软标签：若 objective=min_cost，阈值内按 1/cost 归一
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
            # 回退为映射 one-hot
            name = self.mapper.mapping[c]
            y[i, self.model_names.index(name)] = 1.0

        return BranchResult(labels=labels, representation=X, soft_labels=y)

