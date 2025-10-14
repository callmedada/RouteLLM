from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Optional

from .train_eval import train_and_eval_from_routerbench
from .core import TrainConfig


def grid_search_routerbench(
    routerbench_path: str,
    lambda_list: Iterable[float],
    threshold_list: Iterable[float],
    fuser_types: Iterable[str] = ("mlp", "attention"),
    num_clusters_list: Iterable[int] = (4,),
    fusion_betas: Optional[Iterable[float]] = None,
    limbo_taus: Optional[Iterable[Optional[float]]] = None,
    val_ratio: float = 0.2,
) -> List[Tuple[Dict[str, float], TrainConfig]]:
    results: List[Tuple[Dict[str, float], TrainConfig]] = []
    # 若未显式提供 fusion_betas，则在循环内采用 lambda 作为 fusion_beta；
    # 若未显式提供 limbo_taus，则使用数值默认 0.5（不可为 None）。
    fb_list = list(fusion_betas) if fusion_betas is not None else [0.5]
    tau_list = list(limbo_taus) if limbo_taus is not None else [0.5]
    total = 0
    for _ in num_clusters_list:
        for _ in fuser_types:
            for _ in lambda_list:
                for _ in threshold_list:
                    for _ in fb_list:
                        for _ in tau_list:
                            total += 1
    idx = 0
    for nclu in num_clusters_list:
        for fuser in fuser_types:
            for lam in lambda_list:
                for thr in threshold_list:
                    for fb in fb_list:
                        for tau in tau_list:
                            idx += 1
                            actual_fb = (fb if fb is not None else lam)
                            # 保证 fusion_beta 合法范围 [0,1]
                            try:
                                actual_fb = max(0.0, min(1.0, float(actual_fb)))
                            except Exception:
                                actual_fb = float(lam)
                            actual_tau = tau if tau is not None else 0.5
                            print(
                                f"[grid_search] ({idx}/{total}) n_clusters={nclu}, fuser={fuser}, lambda={lam}, thr={thr}, fusion_beta={actual_fb}, limbo_tau={actual_tau}",
                                flush=True,
                            )
                            cfg = TrainConfig(
                                num_clusters=nclu,
                                beta=0.1,
                                lambda_soft=lam,
                                fusion_beta=actual_fb,
                                objective="min_cost",
                                quality_threshold=thr,
                                limbo_tau=actual_tau,
                                fuser_type=fuser,
                            )
                            metrics = train_and_eval_from_routerbench(
                                routerbench_path, None, None, cfg, val_ratio=val_ratio
                            )
                            results.append((metrics, cfg))
    # 排序：先按 avg_cost 升序，再按 top1_acc 降序
    results.sort(key=lambda x: (x[0].get("avg_cost", 1e9), -x[0].get("top1_acc", 0.0)))
    return results

