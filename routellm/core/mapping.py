from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class ClusterModelMapper:
    def __init__(self, model_names: List[str], model_costs: List[float]) -> None:
        self.model_names: List[str] = list(model_names)
        self.model_costs: np.ndarray = np.asarray(model_costs, dtype=np.float32)
        self.mapping: Dict[int, str] = {}

    def fit(self, labels: List[int], quality: np.ndarray, beta: float) -> Dict[int, str]:
        labels_arr = np.asarray(labels, dtype=np.int32)
        n_clusters = int(labels_arr.max()) + 1 if labels_arr.size > 0 else 0
        self.mapping = {}
        for cluster_id in range(n_clusters):
            indices = [i for i, c in enumerate(labels_arr) if c == cluster_id]
            if len(indices) == 0:
                best_idx = int(np.argmin(self.model_costs))
            else:
                cluster_quality = quality[indices].mean(axis=0)
                utilities = cluster_quality - beta * self.model_costs
                best_idx = int(np.argmax(utilities))
            self.mapping[cluster_id] = self.model_names[best_idx]
        return self.mapping

    def predict(self, cluster_id: int) -> Tuple[str, int]:
        name = self.mapping[cluster_id]
        return name, self.model_names.index(name)

    def fit_min_cost(self, labels: List[int], quality: np.ndarray, quality_threshold: float) -> Dict[int, str]:
        labels_arr = np.asarray(labels, dtype=np.int32)
        n_clusters = int(labels_arr.max()) + 1 if labels_arr.size > 0 else 0
        self.mapping = {}
        for cluster_id in range(n_clusters):
            indices = [i for i, c in enumerate(labels_arr) if c == cluster_id]
            if len(indices) == 0:
                best_idx = int(np.argmin(self.model_costs))
            else:
                cluster_quality = quality[indices].mean(axis=0)
                ok = cluster_quality >= quality_threshold
                if not ok.any():
                    utilities = cluster_quality - 0.0 * self.model_costs
                    best_idx = int(np.argmax(utilities))
                else:
                    costs = np.where(ok, self.model_costs, np.inf)
                    best_idx = int(np.argmin(costs))
            self.mapping[cluster_id] = self.model_names[best_idx]
        return self.mapping

