from __future__ import annotations

from typing import Dict, List, Optional

import json
import os
import numpy as np

from .pipeline import RouterPipeline, TrainConfig
from .data import TrainItem, split_dataset, extract_xy, extract_xy_with_costs, load_routerbench_default, parse_routerbench
from .logger import RunLogger
from .report import generate_evaluation_report


def train_and_eval(
    items: List[TrainItem],
    model_names: List[str],
    model_costs: List[float],
    config: TrainConfig,
    val_ratio: float = 0.2,
) -> Dict[str, float]:
    print(f"[train_and_eval] items={len(items)}, val_ratio={val_ratio}", flush=True)
    train_items, val_items = split_dataset(items, val_ratio=val_ratio)
    tr_texts, tr_feats, tr_labels, tr_q = extract_xy(train_items)
    vl_texts, vl_feats, vl_labels, _, vl_costs = extract_xy_with_costs(val_items)
    print(f"[train_and_eval] split: train={len(tr_texts)}, val={len(vl_texts)}", flush=True)
    quality = None
    if tr_q is not None:
        quality = np.asarray(tr_q, dtype=np.float32)

    print("[train_and_eval] build pipeline", flush=True)
    pipeline = RouterPipeline(model_names, model_costs, config)
    print("[train_and_eval] fitting pipeline", flush=True)
    pipeline.fit(tr_texts, tr_feats, quality=quality)
    print("[train_and_eval] evaluating pipeline", flush=True)
    metrics = pipeline.evaluate(vl_texts, vl_feats, vl_labels, sample_costs=vl_costs)
    # 记录日志
    logger = RunLogger()
    run = logger.create()
    # 将训练配置也写入 metrics.json
    logger.log_metrics(run, metrics, config=config)
    # 生成详细评估报告（若存在标签）
    # 允许部分样本无标签：直接使用 val_items 的 label 列表（含 None）
    try:
        vl_labels_full = [it.label for it in val_items]
        result = generate_evaluation_report(
            pipeline,
            vl_texts,
            vl_feats,
            vl_labels_full,
            model_names,
            model_costs,
            save_dir=str(run.run_dir),
            enable_calibration=config.enable_calibration,
            fallback_strategy=config.fallback_strategy,
            fallback_model_name=config.fallback_model_name,
            per_sample_costs=vl_costs,
        )
        # 在 evaluation.json 中追加 config 字段
        try:
            eval_json_path = os.path.join(str(run.run_dir), "evaluation.json")
            with open(eval_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["config"] = getattr(pipeline, "config", None).__dict__ if hasattr(pipeline, "config") else config.__dict__
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception:
        pass

    try:
        cluster_path = run.run_dir / "limbo_clusters.json"
        cluster_data = pipeline.limbo_branch.collect_cluster_details()
        with cluster_path.open("w", encoding="utf-8") as f:
            json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        try:
            print(f"[train_and_eval] failed to dump LIMBO clusters: {exc}", flush=True)
        except Exception:
            pass
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

