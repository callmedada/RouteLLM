from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .pipeline import RouterPipeline, TrainConfig
from .data import TrainItem, split_dataset, extract_xy, load_routerbench_default, parse_routerbench
from .logger import RunLogger


def train_and_eval(
    items: List[TrainItem],
    model_names: List[str],
    model_costs: List[float],
    config: TrainConfig,
    val_ratio: float = 0.2,
) -> Dict[str, float]:
    train_items, val_items = split_dataset(items, val_ratio=val_ratio)
    tr_texts, tr_feats, tr_labels, tr_q = extract_xy(train_items)
    vl_texts, vl_feats, vl_labels, _ = extract_xy(val_items)
    quality = None
    if tr_q is not None:
        quality = np.asarray(tr_q, dtype=np.float32)

    pipeline = RouterPipeline(model_names, model_costs, config)
    pipeline.fit(tr_texts, tr_feats, quality=quality)
    metrics = pipeline.evaluate(vl_texts, vl_feats, vl_labels)
    # 记录日志
    logger = RunLogger()
    run = logger.create()
    logger.log_metrics(run, metrics)
    return metrics


def train_and_eval_from_routerbench(
    routerbench_path: str,
    model_names: List[str] | None,
    model_costs: List[float] | None,
    config: TrainConfig,
    val_ratio: float = 0.2,
) -> Dict[str, float]:
    items, names, costs = parse_routerbench(routerbench_path)
    model_names = model_names or names
    model_costs = model_costs or costs
    return train_and_eval(items, model_names, model_costs, config, val_ratio=val_ratio)

