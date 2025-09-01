from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .train_eval import train_and_eval_from_routerbench
from .pipeline import TrainConfig


def grid_search_routerbench(
    routerbench_path: str,
    lambda_list: Iterable[float],
    threshold_list: Iterable[float],
    fuser_types: Iterable[str] = ("mlp", "attention"),
    num_clusters_list: Iterable[int] = (4,),
) -> List[Tuple[Dict[str, float], TrainConfig]]:
    results: List[Tuple[Dict[str, float], TrainConfig]] = []
    for nclu in num_clusters_list:
        for fuser in fuser_types:
            for lam in lambda_list:
                for thr in threshold_list:
                    cfg = TrainConfig(
                        num_clusters=nclu,
                        beta=0.1,
                        lambda_soft=lam,
                        objective="min_cost",
                        quality_threshold=thr,
                        fuser_type=fuser,
                    )
                    metrics = train_and_eval_from_routerbench(routerbench_path, None, None, cfg)
                    results.append((metrics, cfg))
    # 排序：先按 avg_cost 升序，再按 top1_acc 降序
    results.sort(key=lambda x: (x[0].get("avg_cost", 1e9), -x[0].get("top1_acc", 0.0)))
    return results

