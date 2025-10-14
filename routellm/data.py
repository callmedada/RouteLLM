from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
import random
import pickle
from pathlib import Path
import pandas as pd


class TrainItem(BaseModel):
    text: str
    features: Dict[str, str] = Field(default_factory=dict)
    label: int | None = None
    quality: List[float] | None = None
    costs: List[float] | None = None


def split_dataset(items: List[TrainItem], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[TrainItem], List[TrainItem]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    k = int(len(shuffled) * (1.0 - val_ratio))
    return shuffled[:k], shuffled[k:]


def extract_xy(items: List[TrainItem]) -> Tuple[List[str], List[Dict[str, str]], list[int] | None, list[list[float]] | None]:
    texts = [it.text for it in items]
    feats = [it.features for it in items]
    labels = [it.label for it in items if it.label is not None]
    labels_out = labels if len(labels) == len(items) else None
    qualities = [it.quality for it in items if it.quality is not None]
    qualities_out = qualities if len(qualities) == len(items) else None
    return texts, feats, labels_out, qualities_out


def extract_xy_with_costs(items: List[TrainItem]) -> Tuple[List[str], List[Dict[str, str]], list[int] | None, list[list[float]] | None, list[list[float]] | None]:
    texts, feats, labels_out, qualities_out = extract_xy(items)
    costs = [it.costs for it in items if it.costs is not None]
    costs_out = costs if len(costs) == len(items) else None
    return texts, feats, labels_out, qualities_out, costs_out


def load_routerbench_default(path: str | Path) -> List[TrainItem]:
    #回 items（costs and names: parse_routerbench_meta）
    items, _, _ = parse_routerbench(path)
    return items


def parse_routerbench(path: str | Path) -> tuple[List[TrainItem], list[str], list[float]]:
    p = Path(path)
    df: pd.DataFrame = pd.read_pickle(p)
    cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
    model_base_names = [c.split("|", 1)[0] for c in cost_cols]
    quality_cols = [c for c in model_base_names if c in df.columns]
    mean_costs = [float(df[f"{m}|total_cost"].fillna(0.0).mean()) for m in model_base_names]

    items: List[TrainItem] = []
    for _, row in df.iterrows():
        text = str(row.get("prompt", ""))
        feats: Dict[str, str] = {"eval": str(row.get("eval_name", "unknown"))}
        q = [float(row.get(m, 0.0)) for m in model_base_names]
        c = [float(row.get(f"{m}|total_cost", 0.0)) for m in model_base_names]
        oracle = row.get("oracle_model_to_route_to")
        label = None
        if isinstance(oracle, str) and oracle in model_base_names:
            label = model_base_names.index(oracle)
        items.append(TrainItem(text=text, features=feats, quality=q, label=label, costs=c))

    return items, model_base_names, mean_costs

