from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LimboAgglomerative:
    """
    极简占位实现，仅用于测试与无外部依赖环境。
    - 对输入特征做一个稳定哈希，将样本均匀分配到 n_clusters。
    - 提供与真实实现一致的 API: fit(features), labels_ 属性。
    """

    n_clusters: int = 2

    def __post_init__(self) -> None:
        self.labels_: List[int] = []

    def fit(self, features: List[Dict[str, str]]) -> "LimboAgglomerative":
        labels: List[int] = []
        for i, f in enumerate(features):
            # 将字典kv对排序拼接做稳定哈希，避免运行间不稳定
            items = sorted((k, v) for k, v in f.items())
            s = "|".join(f"{k}={v}" for k, v in items)
            h = abs(hash(s))
            c = h % max(1, self.n_clusters)
            labels.append(int(c))
        self.labels_ = labels
        return self


