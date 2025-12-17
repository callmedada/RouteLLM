from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import math
import os

import numpy as np


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    s = ex.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return ex / s


def _calc_confusion_and_prf(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels.tolist(), preds.tolist()):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y, p] += 1
    per_model: Dict[int, Dict[str, float]] = {}
    for k in range(num_classes):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - tp)
        fn = float(cm[k, :].sum() - tp)
        precision = tp / max(1.0, (tp + fp))
        recall = tp / max(1.0, (tp + fn))
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_model[k] = {"precision": precision, "recall": recall, "f1": f1, "support": float(cm[k, :].sum())}
    return cm, per_model


def _calc_calibration(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> Dict[str, object]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_conf: List[float] = []
    bin_acc: List[float] = []
    ece = 0.0
    mce = 0.0
    n = max(1, len(conf))
    for b in range(num_bins):
        mask = bin_ids == b
        if not mask.any():
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        mean_conf = float(conf[mask].mean())
        mean_acc = float(acc[mask].mean())
        bin_conf.append(mean_conf)
        bin_acc.append(mean_acc)
        w = float(mask.sum()) / n
        ece += w * abs(mean_conf - mean_acc)
        mce = max(mce, abs(mean_conf - mean_acc))
    return {"bin_confidence": bin_conf, "bin_accuracy": bin_acc, "ECE": float(ece), "MCE": float(mce)}


def _temperature_scale(log_probs: np.ndarray, labels: np.ndarray) -> float:
    """拟合温度 T（标量），使用 log 概率近似为中心化 logits 进行最小化 NLL。
    返回拟合的 T（>=1e-3）。
    """
    # 近似 logits：去均值的 log_probs，因 softmax 对加性常数不敏感
    z = log_probs - log_probs.mean(axis=1, keepdims=True)
    T = 1.0
    lr = 0.1
    for _ in range(100):
        zT = z / max(T, 1e-3)
        p = _safe_softmax(zT)
        # NLL 与导数（对 T）
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(len(labels)), labels] = 1.0
        # d/dT of z/T is -(z/T^2)
        grad = ((p - y_onehot) * (-z) / (max(T, 1e-3) ** 2)).sum()
        T_new = max(1e-3, T - lr * grad / max(1.0, len(labels)))
        if abs(T_new - T) < 1e-5:
            break
        T = T_new
        lr *= 0.95
    return float(max(T, 1e-3))


def _maybe_plot_confusion(cm: np.ndarray, labels: List[str], save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(4, len(labels) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _maybe_plot_calibration(bin_conf: List[float], bin_acc: List[float], save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    xs = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, xs, "--", color="gray", label="perfect")
    ax.plot(bin_conf, bin_acc, "o-", label="empirical")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _maybe_plot_frontier(points: List[Tuple[float, float, float]], save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    # points: (tau, acc, cost)
    xs = [c for (_, _, c) in points]
    ys = [a for (_, a, _) in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", label="frontier")
    ax.set_xlabel("Average Cost")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def generate_evaluation_report(
    pipeline,  # RouterPipeline
    texts: List[str],
    features: List[Dict[str, str]],
    labels: List[Optional[int]],
    model_names: List[str],
    model_costs: List[float],
    save_dir: str,
    *,
    enable_calibration: bool = True,
    fallback_strategy: str = "best_fixed",
    fallback_model_name: Optional[str] = None,
    per_sample_costs: Optional[List[List[float]]] = None,
    quality: Optional[List[List[float]]] = None,
) -> Dict[str, object]:
    os.makedirs(save_dir, exist_ok=True)

    import torch  # local import

    # 批量前向，取出 log 概率、概率、预测与置信度
    x_limbo = pipeline.limbo_branch.vectorizer.transform(features)
    x_bert = pipeline.bert_branch.transform_texts(texts, show_progress_bar=False)
    xn = np.hstack([x_limbo, x_bert])
    # 门控缩放与 pipeline 保持一致
    fusion_beta = float(getattr(pipeline.config, "fusion_beta", pipeline.config.lambda_soft))
    fusion_beta = max(0.0, min(1.0, fusion_beta))
    limbo_dim = x_limbo.shape[1]
    bert_dim = x_bert.shape[1]
    if xn.size > 0 and (limbo_dim + bert_dim) == xn.shape[1]:
        xn = xn.copy()
        xn[:, :limbo_dim] *= fusion_beta
        xn[:, limbo_dim:] *= (1.0 - fusion_beta)
    xn = pipeline._normalize(xn)
    probs: List[np.ndarray] = []
    logps: List[np.ndarray] = []
    preds: List[int] = []
    confs: List[float] = []
    with torch.no_grad():
        for i in range(len(texts)):
            lp = pipeline.fuser_torch(torch.from_numpy(xn[i:i+1].astype(np.float32)))  # type: ignore[attr-defined]
            lp_np = lp.numpy()[0]
            p_np = np.exp(lp_np)
            logps.append(lp_np)
            probs.append(p_np)
            preds.append(int(np.argmax(p_np)))
            confs.append(float(np.max(p_np)))
    probs_np_all = np.asarray(probs, dtype=np.float32)
    logps_np_all = np.asarray(logps, dtype=np.float32)
    preds_np_all = np.asarray(preds, dtype=np.int64)
    # 对齐到有标签的子集
    labels_arr = np.asarray(labels, dtype=object)
    mask = np.array([isinstance(y, (int, np.integer)) for y in labels_arr], dtype=bool)
    labels_np = labels_arr[mask].astype(np.int64)
    probs_np = probs_np_all[mask]
    logps_np = logps_np_all[mask]
    preds_np = preds_np_all[mask]

    # overall cost & acc
    costs_arr = np.asarray(model_costs, dtype=np.float32)
    if per_sample_costs is not None and len(per_sample_costs) == len(texts):
        costs_mat = np.asarray(per_sample_costs, dtype=np.float32)
        avg_cost = float(costs_mat[np.arange(costs_mat.shape[0])[mask], preds_np].mean()) if mask.any() else float(costs_mat[np.arange(costs_mat.shape[0]), preds_np].mean())
    else:
        avg_cost = float(costs_arr[preds_np].mean()) if len(preds_np) else 0.0
    top1_acc = float((preds_np == labels_np).mean()) if len(labels_np) else 0.0
    cost_per_correct = avg_cost / max(1e-12, top1_acc)

    # Extended Metrics
    ext_metrics: Dict[str, float] = {}
    
    # Quality Metrics
    if quality is not None and len(quality) == len(preds_np):
        q_arr = np.asarray(quality, dtype=np.float32)
        # Filter to labeled subset if needed, assuming quality aligns with texts/preds
        # Note: preds_np is filtered by mask (labeled only). 
        # quality input corresponds to all texts.
        # So we need to filter quality by mask.
        if len(quality) == len(texts):
            q_arr = q_arr[mask]
        
        if len(q_arr) == len(preds_np):
            pred_q = q_arr[np.arange(len(preds_np)), preds_np]
            avg_q = float(pred_q.mean())
            oracle_q = float(q_arr.max(axis=1).mean())
            ext_metrics["avg_quality"] = avg_q
            ext_metrics["oracle_quality"] = oracle_q
            ext_metrics["quality_regret"] = oracle_q - avg_q
            ext_metrics["quality_retention"] = avg_q / oracle_q if oracle_q > 1e-9 else 1.0

    # Cost Savings
    max_cost = float(np.max(costs_arr)) if len(costs_arr) > 0 else 0.0
    ext_metrics["cost_savings"] = (max_cost - avg_cost) / max_cost if max_cost > 1e-9 else 0.0

    # Advanced Classification
    if len(labels_np) > 0:
        # Top-k
        for k in [3, 5]:
            if k <= len(model_names):
                top_k_idx = np.argsort(probs_np, axis=1)[:, -k:]
                top_k_hits = np.any(top_k_idx == labels_np[:, None], axis=1).mean()
                ext_metrics[f"top{k}_acc"] = float(top_k_hits)
        
        # F1 & MCC
        try:
            from sklearn.metrics import f1_score, matthews_corrcoef # type: ignore
            ext_metrics["macro_f1"] = float(f1_score(labels_np, preds_np, average="macro"))
            ext_metrics["weighted_f1"] = float(f1_score(labels_np, preds_np, average="weighted"))
            ext_metrics["mcc"] = float(matthews_corrcoef(labels_np, preds_np))
        except (ImportError, Exception):
            pass

        # KL Divergence
        n_classes = len(model_names)
        total_labeled = len(labels_np)
        p_hist = np.bincount(preds_np, minlength=n_classes).astype(np.float32) / max(1, total_labeled)
        q_hist = np.bincount(labels_np, minlength=n_classes).astype(np.float32) / max(1, total_labeled)
        epsilon = 1e-9
        p_hist = np.clip(p_hist, epsilon, 1.0)
        q_hist = np.clip(q_hist, epsilon, 1.0)
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        kl = np.sum(p_hist * np.log(p_hist / q_hist))
        ext_metrics["kl_div"] = float(kl)

    # per-model metrics & confusion matrix
    cm, per_model_prf = _calc_confusion_and_prf(labels_np, preds_np, num_classes=len(model_names))
    route_hist = {name: float((preds_np == i).mean()) for i, name in enumerate(model_names)}

    # calibration
    calib = _calc_calibration(probs_np, labels_np)
    T = 1.0
    if enable_calibration:
        # 用中心化 log 概率作为 logits 近似做温度标定
        T = _temperature_scale(logps_np, labels_np)
    zT = (logps_np - logps_np.mean(axis=1, keepdims=True)) / max(T, 1e-3)
    probs_T = _safe_softmax(zT)
    calib_T = _calc_calibration(probs_T, labels_np)

    # baselines: fixed model acc/cost
    fixed_baselines: Dict[str, Dict[str, float]] = {}
    best_fixed_idx = 0
    best_fixed_acc = -1.0
    for i, name in enumerate(model_names):
        acc_i = float((labels_np == i).mean())
        if per_sample_costs is not None and len(per_sample_costs) == len(texts):
            cost_i = float(np.asarray(per_sample_costs, dtype=np.float32)[:, i].mean())
        else:
            cost_i = float(costs_arr[i])
        fixed_baselines[name] = {"acc": acc_i, "cost": cost_i}
        if acc_i > best_fixed_acc:
            best_fixed_acc = acc_i
            best_fixed_idx = i
    best_fixed_name = model_names[best_fixed_idx]

    # oracle cheapest-correct theoretical lower bound
    oracle_costs: List[float] = []
    for i in range(len(labels_np)):
        y = labels_np[i]
        if per_sample_costs is not None and len(per_sample_costs) == len(texts):
            oracle_costs.append(float(np.asarray(per_sample_costs, dtype=np.float32)[i, y]))
        else:
            oracle_costs.append(float(costs_arr[y]))
    oracle_lower_cost = float(np.mean(oracle_costs)) if oracle_costs else 0.0

    # cost-performance frontier with fallback
    if fallback_strategy == "best_fixed":
        fallback_idx = best_fixed_idx
    elif fallback_strategy == "cheapest":
        fallback_idx = int(np.argmin(costs_arr))
    else:
        fallback_idx = model_names.index(fallback_model_name) if fallback_model_name in model_names else best_fixed_idx

    frontier: List[Tuple[float, float, float]] = []
    taus = np.linspace(0.0, 1.0, 51)
    for tau in taus:
        use_pred = (probs_T.max(axis=1) >= tau)
        chosen = np.where(use_pred, preds_np, fallback_idx)
        acc_tau = float((chosen == labels_np).mean())
        if per_sample_costs is not None and len(per_sample_costs) == len(texts):
            costs_mat = np.asarray(per_sample_costs, dtype=np.float32)
            cost_tau = float(costs_mat[np.arange(costs_mat.shape[0])[mask], chosen[mask]].mean()) if mask.any() else float(costs_mat[np.arange(costs_mat.shape[0]), chosen].mean())
        else:
            cost_tau = float(costs_arr[chosen].mean())
        frontier.append((float(tau), acc_tau, cost_tau))

    # find min-cost point with acc >= current acc
    candidates = [(t, a, c) for (t, a, c) in frontier if a >= top1_acc - 1e-12]
    if candidates:
        best_tau, best_acc, best_cost = min(candidates, key=lambda x: x[2])
        savings_vs_current = (avg_cost - best_cost) / max(1e-12, avg_cost)
    else:
        best_tau, best_acc, best_cost, savings_vs_current = 0.0, top1_acc, avg_cost, 0.0

    # subgroup by eval_name
    groups: Dict[str, Dict[str, float]] = {}
    eval_keys = np.array([f.get("eval", "unknown") for f in features], dtype=object)
    unique_groups = sorted(set(eval_keys.tolist()))
    labeled_indices = np.where(mask)[0]
    # 原始索引 -> 压缩后（仅有标签子集）的位置
    pos_map = {int(orig): int(i) for i, orig in enumerate(labeled_indices.tolist())}
    for g in unique_groups:
        idx_all = np.where(eval_keys == g)[0]
        # 仅取有标签的样本，并映射到压缩后的索引空间
        idx_comp = [pos_map[i] for i in idx_all.tolist() if i in pos_map]
        idx = np.asarray(idx_comp, dtype=np.int64)
        if idx.size == 0:
            continue
        g_acc = float((preds_np[idx] == labels_np[idx]).mean())
        g_cost = float(costs_arr[preds_np[idx]].mean())
        g_routes = {name: float((preds_np[idx] == i).mean()) for i, name in enumerate(model_names)}
        groups[g] = {"top1_acc": g_acc, "avg_cost": g_cost, **{f"route_{k}": v for k, v in g_routes.items()}}

    # assemble json 
    result: Dict[str, object] = {
        "overall": {
            "avg_cost": avg_cost,
            "top1_acc": top1_acc,
            "cost_per_correct": cost_per_correct,
            **ext_metrics,
        },
        "route_ratio": {f"route_{k}": v for k, v in route_hist.items()},
        "per_model": {
            model_names[i]: {
                **per_model_prf[i],
                "model_cost": float(costs_arr[i]),
                "routed_ratio": float(route_hist[model_names[i]]),
            }
            for i in range(len(model_names))
        },
        "confusion_matrix": {
            "labels": model_names,
            "matrix": cm.astype(int).tolist(),
        },
        "calibration": calib,
        "calibration_after_temperature": {**calib_T, "T": T},
        "baselines": {
            "fixed": fixed_baselines,
            "best_fixed": {"name": best_fixed_name, "acc": best_fixed_acc, "cost": float(costs_arr[best_fixed_idx])},
            "oracle_cheapest_correct_cost": oracle_lower_cost,
        },
        "frontier": {
            "points": [(float(t), float(a), float(c)) for (t, a, c) in frontier],
            "best_under_current_acc": {"tau": best_tau, "acc": best_acc, "cost": best_cost, "savings_vs_current": savings_vs_current},
            "fallback": {"strategy": fallback_strategy, "model_index": int(fallback_idx), "model_name": model_names[fallback_idx]},
        },
        "subgroups": groups,
    }

    # write json
    json_path = os.path.join(save_dir, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # write markdown
    md_path = os.path.join(save_dir, "evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"- avg_cost: {avg_cost:.6f}\n")
        f.write(f"- top1_acc: {top1_acc:.4f}\n")
        f.write(f"- cost_per_correct: {cost_per_correct:.6f}\n")
        if "cost_savings" in ext_metrics:
            f.write(f"- cost_savings: {ext_metrics['cost_savings']:.2%}\n")
        
        if "avg_quality" in ext_metrics:
            f.write("\n### Quality Analysis\n")
            f.write(f"- avg_quality: {ext_metrics['avg_quality']:.4f}\n")
            f.write(f"- oracle_quality: {ext_metrics['oracle_quality']:.4f}\n")
            f.write(f"- quality_regret: {ext_metrics['quality_regret']:.4f}\n")
            f.write(f"- quality_retention: {ext_metrics['quality_retention']:.2%}\n")
        
        if "macro_f1" in ext_metrics or "top3_acc" in ext_metrics:
            f.write("\n### Advanced Classification\n")
            if "top3_acc" in ext_metrics:
                f.write(f"- top3_acc: {ext_metrics['top3_acc']:.4f}\n")
            if "top5_acc" in ext_metrics:
                f.write(f"- top5_acc: {ext_metrics['top5_acc']:.4f}\n")
            if "macro_f1" in ext_metrics:
                f.write(f"- macro_f1: {ext_metrics['macro_f1']:.4f}\n")
            if "weighted_f1" in ext_metrics:
                f.write(f"- weighted_f1: {ext_metrics['weighted_f1']:.4f}\n")
            if "mcc" in ext_metrics:
                f.write(f"- mcc: {ext_metrics['mcc']:.4f}\n")
            if "kl_div" in ext_metrics:
                f.write(f"- kl_div: {ext_metrics['kl_div']:.4f}\n")
        f.write("\n")

        # 记录关键超参数（含融合与LIMBO）
        cfg = getattr(pipeline, "config", None)
        if cfg is not None:
            fusion_beta_md = getattr(cfg, "fusion_beta", getattr(cfg, "lambda_soft", None))
            limbo_tau_md = getattr(cfg, "limbo_tau", None)
            utility_beta_md = getattr(cfg, "utility_beta", getattr(cfg, "beta", None))
            f.write("## Hyperparameters\n\n")
            f.write(f"- fusion_beta: {fusion_beta_md}\n")
            f.write(f"- limbo_tau: {limbo_tau_md}\n")
            f.write(f"- utility_beta: {utility_beta_md}\n")
            f.write(f"- frontier_best_tau: {best_tau}\n\n")
        f.write("## Route Ratios\n\n")
        for k, v in route_hist.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Per-model (precision/recall/f1)\n\n")
        for i, name in enumerate(model_names):
            pm = per_model_prf[i]
            f.write(f"- {name}: precision={pm['precision']:.4f}, recall={pm['recall']:.4f}, f1={pm['f1']:.4f}, support={pm['support']:.0f}, cost={float(costs_arr[i])}\n")
        f.write("\n## Calibration\n\n")
        f.write(f"- ECE: {calib['ECE']}  MCE: {calib['MCE']}\n")
        f.write(f"- After T={T:.3f}, ECE: {calib_T['ECE']}  MCE: {calib_T['MCE']}\n\n")
        f.write("## Baselines\n\n")
        f.write(f"- Best fixed: {best_fixed_name} (acc={best_fixed_acc}, cost={float(costs_arr[best_fixed_idx])})\n")
        f.write(f"- Oracle cheapest-correct cost: {oracle_lower_cost}\n\n")
        f.write("## Frontier\n\n")
        f.write(f"- Best under current acc: tau={best_tau}, acc={best_acc}, cost={best_cost}, savings_vs_current={savings_vs_current}\n\n")
        if groups:
            f.write("## Subgroups by eval\n\n")
            for g, stats in groups.items():
                f.write(f"- {g}: top1_acc={stats['top1_acc']}, avg_cost={stats['avg_cost']}\n")

    # plots
    _maybe_plot_confusion(cm, model_names, os.path.join(save_dir, "confusion_matrix.png"))
    _maybe_plot_calibration(calib["bin_confidence"], calib["bin_accuracy"], os.path.join(save_dir, "calibration.png"))
    _maybe_plot_frontier(result["frontier"]["points"], os.path.join(save_dir, "cost_frontier.png"))

    return result


