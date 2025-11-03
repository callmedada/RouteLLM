from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..encoders import build_text_encoder


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

    def fit(self, texts: List[str], quality: np.ndarray) -> BranchResult:
        # 大批量时让底层 encoder 显示进度，小批量不显示，避免条目不连贯
        try:
            print(f"[BertBranch.fit] start: n_texts={len(texts)}, prefer_transformer={hasattr(self.encoder, '_model')}", flush=True)
        except Exception:
            pass
        X = self.encoder.encode(texts, show_progress_bar=len(texts) >= 32)
        if X.size == 0:
            Xn = X
        else:
            min_v = X.min(axis=0, keepdims=True)
            max_v = X.max(axis=0, keepdims=True)
            denom = max_v - min_v
            denom[denom == 0] = 1.0
            self.norm_min = min_v
            self.norm_max = max_v
            Xn = (X - min_v) / denom
        try:
            print(f"[BertBranch.fit] embeddings ready: shape={X.shape}, normed_shape={Xn.shape}", flush=True)
        except Exception:
            pass

        labels = list(range(len(texts)))
        y = self._build_soft_labels(quality)
        return BranchResult(labels=labels, representation=Xn, soft_labels=y)

    def _build_soft_labels(self, quality: np.ndarray) -> np.ndarray:
        if quality.ndim != 2 or quality.shape[1] != len(self.model_names):
            raise ValueError("quality 矩阵形状不匹配 BERT 分支需要的维度")
        n, m = quality.shape
        y = np.zeros((n, m), dtype=np.float32)
        if self.objective == "min_cost":
            costs = np.asarray(self.model_costs, dtype=np.float32)
            for i in range(n):
                q = quality[i]
                ok = q >= self.quality_threshold
                if ok.any():
                    inv_cost = np.where(ok, 1.0 / costs, 0.0)
                    s = inv_cost.sum()
                    if s > 0:
                        y[i] = inv_cost / s
                        continue
                best_idx = int(np.argmin(costs))
                y[i, best_idx] = 1.0
            return y

        # objective == utility 或其他默认：按质量选最优
        for i in range(n):
            row = quality[i]
            if np.all(row == row[0]):
                y[i, :] = 1.0 / m
                continue
            best = int(np.argmax(row))
            y[i, best] = 1.0
        return y

    def transform_texts(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        try:
            print(f"[BertBranch.transform_texts] n={len(texts)}", flush=True)
        except Exception:
            pass
        X_raw = self.encoder.encode(texts, show_progress_bar=show_progress_bar)
        if X_raw.size == 0:
            return X_raw
        if self.norm_min is not None and self.norm_max is not None:
            denom = self.norm_max - self.norm_min
            denom = np.where(denom == 0, 1.0, denom)
            return (X_raw - self.norm_min) / denom
        if X_raw.shape[0] <= 1:
            return X_raw
        min_v = X_raw.min(axis=0, keepdims=True)
        max_v = X_raw.max(axis=0, keepdims=True)
        denom = max_v - min_v
        denom[denom == 0] = 1.0
        return (X_raw - min_v) / denom

