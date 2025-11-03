from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..encoders import CategoricalFeatureVectorizer
from ..clusterers import LoggedCoarseToLimboClusterer
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
class BranchResult:
    labels: List[int]
    representation: np.ndarray
    soft_labels: np.ndarray


class LimboBranch:
    """
    LIMBO
    """

    def __init__(
        self,
        model_names: List[str],
        model_costs: List[float],
        num_clusters: int,
        beta: float,
        objective: str = "utility",
        quality_threshold: float = 0.0,
        *,
        limbo_tau: float | None = None,
        limbo_use_sparse: bool = True,
        input_dump_path: str | None = None,
        coarse_clusters: int | None = None,
        limbo_fit_size: int = 2000,
        prob_temp: float = 1.0,
        random_state: int = 42,
        progress_logging: bool = True,
    ) -> None:
        self.model_names = model_names
        self.model_costs = model_costs
        self.num_clusters = max(1, num_clusters)
        self.beta = beta
        self.objective = objective
        self.quality_threshold = quality_threshold

        # Coarse-to-LIMBO 聚类器
        inferred_coarse = max(self.num_clusters * 4, self.num_clusters + 4)
        self.coarse_clusters = int(coarse_clusters) if coarse_clusters is not None else inferred_coarse
        self.limbo_fit_size = int(limbo_fit_size)
        self.prob_temp = float(prob_temp)
        self.random_state = int(random_state)
        self.clusterer = LoggedCoarseToLimboClusterer(
            model_names=model_names,
            model_costs=model_costs,
            beta=beta,
            coarse_clusters=self.coarse_clusters,
            limbo_clusters=self.num_clusters,
            limbo_tau=limbo_tau,
            limbo_fit_size=self.limbo_fit_size,
            prob_temp=self.prob_temp,
            random_state=self.random_state,
            enable_progress_logging=progress_logging,
        )
        self._limbo_use_sparse = bool(limbo_use_sparse)
        self.vectorizer = CategoricalFeatureVectorizer()
        self.norm_min: np.ndarray | None = None
        self.norm_max: np.ndarray | None = None
        self._input_dump_path: Path | None = Path(input_dump_path) if input_dump_path else None

    def _dump_inputs(self, features: List[Dict[str, str]], quality: np.ndarray | None) -> None:
        if self._input_dump_path is None:
            return
        try:
            path = self._input_dump_path
            if path.suffix == "":
                path = path / "limbo_inputs.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            serial_features: List[Dict[str, str]] = []
            for item in features:
                if not item:
                    serial_features.append({})
                    continue
                serial_features.append({str(k): "" if v is None else str(v) for k, v in item.items()})
            payload: Dict[str, object] = {
                "model_names": list(self.model_names),
                "model_costs": [float(x) for x in self.model_costs],
                "features": serial_features,
            }
            if quality is not None:
                payload["quality"] = quality.astype(np.float32).tolist()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            try:
                print(f"[LimboBranch.fit] inputs dumped to {path}", flush=True)
            except Exception:
                pass
        except Exception as exc:
            try:
                print(f"[LimboBranch.fit] failed to dump inputs: {exc}", flush=True)
            except Exception:
                pass

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        min_v = x.min(axis=0, keepdims=True)
        max_v = x.max(axis=0, keepdims=True)
        denom = max_v - min_v
        denom[denom == 0] = 1.0
        return (x - min_v) / denom

    def fit(self, features: List[Dict[str, str]], quality: np.ndarray) -> BranchResult:
        # 聚类
        try:
            print(f"[LimboBranch.fit] start: n={len(features)}, num_clusters={self.num_clusters}", flush=True)
        except Exception:
            pass
        n_samples = len(features)
        coarse_current = getattr(self.clusterer, "coarse_clusters", None)
        if isinstance(coarse_current, int) and coarse_current > n_samples and n_samples > 0:
            adjusted = max(1, n_samples)
            self.clusterer.coarse_clusters = adjusted
            self.coarse_clusters = adjusted
            try:
                print(
                    f"[LimboBranch.fit] adjusted coarse_clusters from {coarse_current} to {adjusted} (n_samples={n_samples})",
                    flush=True,
                )
            except Exception:
                pass
        self._dump_inputs(features, quality)
        self.clusterer.set_quality(quality)
        self.clusterer.fit(features)
        labels_arr = np.asarray(self.clusterer.labels_, dtype=np.int32)
        labels = labels_arr.tolist()


        print("[LimboBranch.fit] vectorizing features ...", flush=True)
        self.vectorizer.fit(features)
        X_raw = self.vectorizer.transform(features)
        print("X_raw after transformation, input for limbo", X_raw)
        if X_raw.size == 0:
            X = X_raw
        else:
            min_v = X_raw.min(axis=0, keepdims=True)
            max_v = X_raw.max(axis=0, keepdims=True)
            denom = max_v - min_v
            denom[denom == 0] = 1.0
            self.norm_min = min_v
            self.norm_max = max_v
            X = (X_raw - min_v) / denom
        print(f"[LimboBranch.fit] normed feature shape={X.shape}", flush=True)
        cluster_probs = self._resolve_cluster_probs(labels_arr, quality)

        y = np.zeros((len(labels), len(self.model_names)), dtype=np.float32)
        for i, c in enumerate(tqdm(labels, desc="LIMBO branch labels", leave=False)):
            prob = cluster_probs.get(int(c))
            if prob is None:
                prob = np.zeros(len(self.model_names), dtype=np.float32)
                if self.model_names:
                    prob[0] = 1.0
            y[i] = prob.astype(np.float32)
        print(f"[LimboBranch.fit] done: labels={len(labels)}, y_shape={y.shape}", flush=True)
        return BranchResult(labels=labels, representation=X, soft_labels=y)

    def _resolve_cluster_probs(self, labels: np.ndarray, quality: np.ndarray) -> Dict[int, np.ndarray]:
        cluster_probs: Dict[int, np.ndarray] = {}
        existing_probs = getattr(self.clusterer, "cluster_model_probs_", None)
        if existing_probs:
            cluster_probs = {int(k): np.asarray(v, dtype=np.float32) for k, v in existing_probs.items()}

        if self.objective == "min_cost":
            min_cost = self._cluster_probs_min_cost(labels, quality)
            cluster_probs.update(min_cost)
        else:
            # 确保至少存在 softmax 概率
            if not cluster_probs:
                cluster_probs = self._cluster_probs_utility(labels, quality)

        # 同步 clusterer 属性，便于外部访问
        mapping = {}
        for cid, prob in cluster_probs.items():
            if prob.size == 0:
                continue
            idx = int(np.argmax(prob))
            if 0 <= idx < len(self.model_names):
                mapping[cid] = self.model_names[idx]
        self.clusterer.cluster_model_probs_ = cluster_probs
        self.clusterer.cluster_model_mapping_ = mapping
        return cluster_probs

    def _cluster_probs_utility(self, labels: np.ndarray, quality: np.ndarray) -> Dict[int, np.ndarray]:
        probs: Dict[int, np.ndarray] = {}
        model_costs = np.asarray(self.model_costs, dtype=np.float32)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_quality = quality[mask].mean(axis=0)
            utilities = cluster_quality - self.beta * model_costs
            z = utilities - utilities.max()
            exp = np.exp(z)
            denom = float(exp.sum())
            if denom <= 0:
                denom = 1.0
            probs[int(cluster_id)] = (exp / denom).astype(np.float32)
        return probs

    def _cluster_probs_min_cost(self, labels: np.ndarray, quality: np.ndarray) -> Dict[int, np.ndarray]:
        probs: Dict[int, np.ndarray] = {}
        costs = np.asarray(self.model_costs, dtype=np.float32)
        n_models = len(costs)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_quality = quality[mask].mean(axis=0)
            ok = cluster_quality >= self.quality_threshold
            if ok.any():
                inv_cost = np.where(ok, 1.0 / np.clip(costs, 1e-6, None), 0.0)
                weight = inv_cost.sum()
                if weight > 0:
                    probs[int(cluster_id)] = (inv_cost / weight).astype(np.float32)
                    continue
            utilities = cluster_quality - self.beta * costs
            best = int(np.argmax(utilities)) if n_models else -1
            vec = np.zeros(n_models, dtype=np.float32)
            if 0 <= best < n_models:
                vec[best] = 1.0
            probs[int(cluster_id)] = vec
        return probs

    def summary(self, top_k: int | None = 5) -> Dict[str, object]:
        if self.clusterer.labels_ is None:
            raise RuntimeError("clusterer 未训练，无法生成摘要")
        counts = Counter(int(x) for x in self.clusterer.labels_.tolist())
        probs = getattr(self.clusterer, "cluster_model_probs_", {}) or {}
        mapping = getattr(self.clusterer, "cluster_model_mapping_", {}) or {}
        limit = len(counts)
        if isinstance(top_k, int) and top_k > 0:
            limit = min(top_k, len(counts))
        ranked = []
        for cid, cnt in counts.most_common(max(1, limit)):
            prob = np.asarray(probs.get(cid, np.zeros(len(self.model_names))), dtype=np.float32)
            ranked.append(
                {
                    "cluster_id": int(cid),
                    "count": int(cnt),
                    "best_model": mapping.get(cid),
                    "probs": prob.tolist(),
                }
            )
        return {
            "n_clusters": len(counts),
            "top_clusters": ranked,
            "coarse_clusters": int(getattr(self.clusterer, "coarse_clusters", self.coarse_clusters)),
        }

    def cluster_profiles(self) -> List[Dict[str, object]]:
        if getattr(self.clusterer, "cluster_model_probs_", None) is None:
            return []
        mapping = getattr(self.clusterer, "cluster_model_mapping_", {}) or {}
        probs = getattr(self.clusterer, "cluster_model_probs_", {}) or {}
        profiles: List[Dict[str, object]] = []
        for cid in sorted(probs):
            prob = np.asarray(probs[cid], dtype=np.float32)
            profiles.append(
                {
                    "cluster_id": int(cid),
                    "best_model": mapping.get(cid),
                    "probs": prob.tolist(),
                }
            )
        return profiles

    def collect_cluster_details(self, feature_top_k: int | None = None) -> Dict[str, object]:
        if self.clusterer.labels_ is None:
            raise RuntimeError("clusterer 未训练，无法收集详情")

        labels = np.asarray(self.clusterer.labels_, dtype=np.int32)
        probs = getattr(self.clusterer, "cluster_model_probs_", {}) or {}
        mapping = getattr(self.clusterer, "cluster_model_mapping_", {}) or {}
        info = getattr(self.clusterer, "cluster_info_", {}) or {}
        coarse_labels = getattr(self.clusterer, "coarse_labels_", None)

        cluster_model_probs = {
            int(cid): np.asarray(prob, dtype=np.float32).tolist() for cid, prob in probs.items()
        }

        overall = {
            "labels": labels.tolist(),
            "cluster_counts": {int(k): int(v) for k, v in Counter(labels.tolist()).items()},
            "cluster_model_mapping": {int(k): v for k, v in mapping.items()},
            "cluster_model_probs": cluster_model_probs,
            "summary": self.summary(top_k=None),
            "profiles": self.cluster_profiles(),
        }
        if coarse_labels is not None:
            overall["coarse_labels"] = np.asarray(coarse_labels, dtype=np.int32).tolist()

        coarse_details: List[Dict[str, object]] = []
        limbo_models = getattr(self.clusterer, "_limbo_models", {}) or {}
        dcf_clusters: List[Dict[str, object]] = []
        for coarse_id in sorted(limbo_models):
            limbo, offset = limbo_models[coarse_id]
            info_obj = info.get(coarse_id)
            info_dict: Dict[str, object] = {}
            if info_obj is not None:
                info_dict = {
                    "coarse_size": int(getattr(info_obj, "coarse_size", 0)),
                    "unique_records": int(getattr(info_obj, "unique_records", 0)),
                    "limbo_clusters": int(getattr(info_obj, "limbo_clusters", 0)),
                }

            limbo_summary: Dict[str, object] = {"n_clusters": 0, "clusters": []}
            clusters_detail: List[Dict[str, object]] = []
            if limbo is not None:
                try:
                    k = feature_top_k if feature_top_k is not None else 10
                    limbo_summary_raw = limbo.summary(top_k=k)
                except Exception:
                    limbo_summary_raw = {}
                try:
                    limbo_profiles_raw = limbo.cluster_profiles()
                except Exception:
                    limbo_profiles_raw = []

                sizes = limbo_summary_raw.get("sizes", []) if isinstance(limbo_summary_raw, dict) else []
                entropies = limbo_summary_raw.get("entropy", []) if isinstance(limbo_summary_raw, dict) else []
                top_features = limbo_summary_raw.get("top_features", []) if isinstance(limbo_summary_raw, dict) else []
                separation = limbo_summary_raw.get("separation") if isinstance(limbo_summary_raw, dict) else None
                n_local = max(len(limbo_profiles_raw), len(top_features), len(sizes))

                for local_idx in range(n_local):
                    final_cluster_id = int(offset + local_idx)
                    detail: Dict[str, object] = {
                        "cluster_id": final_cluster_id,
                        "coarse_cluster_id": int(coarse_id),
                        "local_cluster": int(local_idx),
                    }
                    if local_idx < len(sizes):
                        try:
                            detail["size"] = int(sizes[local_idx])
                        except Exception:
                            detail["size"] = sizes[local_idx]
                    if local_idx < len(entropies):
                        try:
                            detail["entropy"] = float(entropies[local_idx])
                        except Exception:
                            detail["entropy"] = entropies[local_idx]
                    if local_idx < len(top_features):
                        detail["top_features"] = top_features[local_idx]
                    if local_idx < len(limbo_profiles_raw):
                        detail["profile"] = limbo_profiles_raw[local_idx]
                    clusters_detail.append(detail)
                    dcf_clusters.append(detail)

                limbo_summary = {
                    "n_clusters": limbo_summary_raw.get("n_clusters") if isinstance(limbo_summary_raw, dict) else len(clusters_detail),
                    "separation": separation,
                    "clusters": clusters_detail,
                }

            coarse_details.append(
                {
                    "coarse_cluster_id": int(coarse_id),
                    "offset": int(offset),
                    "info": info_dict,
                    "limbo_summary": limbo_summary,
                }
            )

        overall["coarse_clusters"] = coarse_details
        overall["dcf_clusters"] = sorted(dcf_clusters, key=lambda x: x.get("cluster_id", 0))
        return overall

    def dump_cluster_details(self, output_path: str | Path, *, feature_top_k: int | None = None) -> None:
        data = self.collect_cluster_details(feature_top_k=feature_top_k)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

