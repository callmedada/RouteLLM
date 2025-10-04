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
    show_progress: bool = True
    # 成本感知训练与校准
    cost_weight: float = 0.5              # alpha：成本项权重
    softmax_temperature: float = 0.5      # tau：softmax 温度
    blend_costaware: float = 0.5          # gamma：与分支软标签的融合权重
    enable_calibration: bool = True       # 是否在验证集做温度校准（报告阶段使用）
    fallback_strategy: str = "best_fixed" # 成本-性能前沿回退策略：best_fixed|cheapest|name
    fallback_model_name: str | None = None

