from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Optional

import sys
from itertools import product

from .train_eval import train_and_eval_from_routerbench
from .core import TrainConfig


try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    _TQDM_AVAILABLE = False

    def tqdm(x, **kwargs):  # type: ignore
        return x


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

    cluster_values = list(num_clusters_list)
    fuser_values = list(fuser_types)
    lambda_values = list(lambda_list)
    threshold_values = list(threshold_list)

    param_iterable = list(
        product(cluster_values, fuser_values, lambda_values, threshold_values, fb_list, tau_list)
    )
    total = len(param_iterable)

    if _TQDM_AVAILABLE:
        iterator = tqdm(param_iterable, desc="grid search", total=total, leave=False)
    else:
        iterator = param_iterable
        print(f"[grid_search] total combinations: {total}", flush=True)

    for idx, (nclu, fuser, lam, thr, fb, tau) in enumerate(iterator, start=1):
        actual_fb = fb if fb is not None else lam
        try:
            actual_fb = max(0.0, min(1.0, float(actual_fb)))
        except Exception:
            actual_fb = float(lam)
        actual_tau = tau if tau is not None else 0.5

        if _TQDM_AVAILABLE and hasattr(iterator, "set_postfix"):
            try:
                iterator.set_postfix({
                    "clusters": nclu,
                    "fuser": fuser,
                    "lambda": f"{lam:.3g}",
                    "thr": f"{thr:.3g}",
                })
            except Exception:
                pass
        elif not _TQDM_AVAILABLE:
            sys.stdout.write(
                f"\r[grid_search] ({idx}/{total}) n_clusters={nclu}, fuser={fuser}, "
                f"lambda={lam}, thr={thr}, fusion_beta={actual_fb}, limbo_tau={actual_tau}"
            )
            sys.stdout.flush()

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

    if not _TQDM_AVAILABLE:
        print()  # 保证输出换行

    # 排序：先按 avg_cost 升序，再按 top1_acc 降序
    results.sort(key=lambda x: (x[0].get("avg_cost", 1e9), -x[0].get("top1_acc", 0.0)))
    return results

