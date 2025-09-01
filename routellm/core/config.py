from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    num_clusters: int
    beta: float
    lambda_soft: float
    prefer_transformer: bool = True
    embedding_dim: int = 384
    fuser_hidden: Tuple[int, int] = (256, 128)
    seed: int = 42
    fuser_epochs: int = 50
    fuser_lr: float = 1e-2
    objective: str = "utility"
    quality_threshold: float = 0.0
    fuser_type: str = "mlp"  # mlp | attention

