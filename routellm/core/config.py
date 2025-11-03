from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    num_clusters: int
    beta: float  # legacy: kept for compatibility; see utility_beta
    lambda_soft: float  # legacy: mapped to fusion_beta if fusion_beta unchanged
    # 新增：分支融合门控（β=0 -> 仅BERT；β=1 -> 仅LIMBO）
    fusion_beta: float = 0.5
    # 新增：LIMBO 聚类器温度/早停阈值（占位实现存储不改变逻辑）
    limbo_tau: Optional[float] = None
    limbo_use_sparse: bool = True
    # 新增：簇内 utility 的成本权重别名（默认等于 legacy beta）
    utility_beta: Optional[float] = None
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
    limbo_input_dump_path: str | None = None
    use_query_features: bool = True
    use_spacy_features: bool = True
    spacy_model: str = "en_core_web_sm"
    spacy_enable_components: Tuple[str, ...] = ("tok2vec", "tagger", "parser", "ner")
    query_hash_buckets: int = 128
    query_max_hash_per_ngram: int = 64
    query_embed_k: Optional[int] = None
    limbo_coarse_clusters: Optional[int] = None
    limbo_fit_size: int = 2000
    limbo_prob_temp: float = 1.0

    def __post_init__(self) -> None:
        # legacy beta -> utility_beta（若未显式提供）
        if self.utility_beta is None:
            self.utility_beta = self.beta
        # legacy lambda_soft -> fusion_beta（仅当 fusion_beta 仍为默认值）
        # 以便旧调用只传 lambda_soft 仍可生效
        try:
            default_fusion_beta = type(self).fusion_beta
        except Exception:
            default_fusion_beta = 0.5
        if (self.fusion_beta == default_fusion_beta) and (self.lambda_soft is not None):
            self.fusion_beta = self.lambda_soft

        if not self.use_spacy_features:
            raise ValueError("use_spacy_features 必须为 True，以确保启用 spaCy 特征")

        if self.query_embed_k is not None and self.query_embed_k <= 0:
            raise ValueError("query_embed_k 必须为正整数或 None")

