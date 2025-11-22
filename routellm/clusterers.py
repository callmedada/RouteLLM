from __future__ import annotations

from collections import Counter
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from limbo_cluster.agglomerative import LimboAgglomerative
from limbo_cluster.coarse import (  # type: ignore
    CoarseToLimboClusterer as _Base,
    ClusterInfo,
    MiniBatchKSDivergence,
)


class LoggedCoarseToLimboClusterer:
    """Two-stage clustering with optional progress logging."""

    def __init__(
        self,
        *,
        model_names: Sequence[str],
        model_costs: Sequence[float],
        quality: np.ndarray | None = None,
        beta: float = 0.3,
        coarse_clusters: int = 150,
        coarse_batch_size: int = 2048,
        coarse_max_iter: int = 200,
        limbo_clusters: int = 12,
        limbo_tau: float | None = None,
        limbo_fit_size: int = 2000,
        prob_temp: float = 1.0,
        random_state: int = 42,
        enable_progress_logging: bool = True,
    ) -> None:
        self._base = _Base(
            model_names=model_names,
            model_costs=model_costs,
            quality=quality,
            beta=beta,
            coarse_clusters=coarse_clusters,
            coarse_batch_size=coarse_batch_size,
            coarse_max_iter=coarse_max_iter,
            limbo_clusters=limbo_clusters,
            limbo_tau=limbo_tau,
            limbo_fit_size=limbo_fit_size,
            prob_temp=prob_temp,
            random_state=random_state,
        )

        self.enable_progress_logging = enable_progress_logging
        self._coarse_clusters = int(coarse_clusters)
        self.coarse_batch_size = coarse_batch_size
        self.coarse_max_iter = coarse_max_iter
        self.limbo_clusters = limbo_clusters
        self.limbo_tau = limbo_tau
        self.limbo_fit_size = limbo_fit_size
        self.prob_temp = prob_temp
        self.random_state = random_state
        self.model_names = list(model_names)
        self.model_costs = np.asarray(model_costs, dtype=float)
        self.beta = float(beta)

        # exposed fields
        self.labels_: np.ndarray | None = None
        self.coarse_labels_: np.ndarray | None = None
        self.cluster_model_mapping_: Dict[int, str] | None = None
        self.cluster_model_probs_: Dict[int, np.ndarray] | None = None
        self.cluster_info_: Dict[int, ClusterInfo] | None = None
        self._limbo_models: Dict[int, Tuple[LimboAgglomerative, int]] = {}
        self._vectorizer: DictVectorizer | None = None
        self._coarse_model: MiniBatchKSDivergence | None = None
        self.coarse_summary_: Dict[str, Any] | None = None

    @property
    def coarse_clusters(self) -> int:
        return int(getattr(self._base, "coarse_clusters", self._coarse_clusters))

    @coarse_clusters.setter
    def coarse_clusters(self, value: int) -> None:
        self._coarse_clusters = int(value)
        self._base.coarse_clusters = int(value)

    def set_quality(self, quality: np.ndarray) -> None:
        self._base.set_quality(quality)

    def fit(self, records: Sequence[Dict[str, str]], *, quality: np.ndarray | None = None) -> "LoggedCoarseToLimboClusterer":
        if quality is not None:
            self._base.set_quality(quality)

        base = self._base
        if base.quality is None:
            raise ValueError("Quality matrix must be provided before fitting")

        records = list(records)
        n_records = len(records)
        if n_records == 0:
            raise ValueError("records cannot be empty")
        if base.quality.shape[0] != n_records:
            raise ValueError("quality rows must match number of records")
        if base.quality.shape[1] != len(self.model_names):
            raise ValueError("quality columns must match number of models")

        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(records)
        X = self._ensure_csr(X)

        coarse_model = MiniBatchKSDivergence(
            n_clusters=base.coarse_clusters,
            random_state=base.random_state,
            batch_size=base.coarse_batch_size,
            max_iter=base.coarse_max_iter,
            assign_batch_size=max(1024, base.coarse_batch_size),
        )
        feature_names = vectorizer.get_feature_names_out()

        if self.enable_progress_logging:
            print(
                "[CoarseToLimboClusterer.fit] starting coarse stage with MiniBatch KS divergence",
                flush=True,
            )

        coarse_labels = coarse_model.fit_predict(X, feature_names=feature_names)
        self.coarse_summary_ = coarse_model.summary()

        final_labels = np.empty(n_records, dtype=int)
        cluster_info: Dict[int, ClusterInfo] = {}
        limbo_models: Dict[int, Tuple[LimboAgglomerative, int]] = {}
        current_offset = 0
        rng = np.random.default_rng(base.random_state)

        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(coarse_labels):
            buckets[int(label)].append(idx)

        total_buckets = len(buckets)
        if self.enable_progress_logging:
            print(
                f"[CoarseToLimboClusterer.fit] total coarse clusters={total_buckets}",
                flush=True,
            )

        for progress_idx, coarse_id in enumerate(sorted(buckets), start=1):
            idx_list = buckets[coarse_id]
            unique_records, uid_to_indices = base._deduplicate_records(idx_list, records)
            if self.enable_progress_logging:
                print(
                    f"[CoarseToLimboClusterer.fit] processing coarse {progress_idx}/{total_buckets} "
                    f"(id={coarse_id}, size={len(idx_list)}, unique={len(unique_records)})",
                    flush=True,
                )
            limbo, unique_labels = base._fit_limbo_with_cap(unique_records, rng)
            label_map = np.asarray(unique_labels, dtype=int)

            local_unique = len(set(label_map.tolist()))
            for uid, orig_indices in uid_to_indices.items():
                final_cluster = current_offset + label_map[uid]
                for original_idx in orig_indices:
                    final_labels[original_idx] = final_cluster

            cluster_info[coarse_id] = ClusterInfo(
                coarse_size=len(idx_list),
                unique_records=len(unique_records),
                limbo_clusters=local_unique,
            )
            limbo_models[coarse_id] = (limbo, current_offset)
            current_offset += local_unique

        self.labels_ = final_labels
        self.cluster_info_ = cluster_info
        self._limbo_models = limbo_models
        self.coarse_labels_ = coarse_labels

        mapping: Dict[int, str] = {}
        probabilities: Dict[int, np.ndarray] = {}
        for cluster_id in np.unique(final_labels):
            mask = final_labels == cluster_id
            cluster_quality = base.quality[mask].mean(axis=0)
            utilities = cluster_quality - base.beta * self.model_costs
            best_idx = int(np.argmax(utilities))
            mapping[int(cluster_id)] = self.model_names[best_idx]
            probabilities[int(cluster_id)] = base._softmax(utilities, base.prob_temp)

        self.cluster_model_mapping_ = mapping
        self.cluster_model_probs_ = probabilities

        self._vectorizer = vectorizer
        self._coarse_model = coarse_model

        if self.enable_progress_logging:
            summary = self.coarse_summary_ or {}
            n_coarse = summary.get("n_clusters")
            print(
                f"[CoarseToLimboClusterer.fit] coarse summary ready (n_clusters={n_coarse})",
                flush=True,
            )

        return self

    def predict(self, records: Sequence[Dict[str, str]]) -> np.ndarray:
        if not records:
            raise ValueError("records cannot be empty")
        if self._vectorizer is None or self._coarse_model is None:
            raise RuntimeError("You must fit the clusterer before calling predict")

        X = self._vectorizer.transform(records)
        X = self._ensure_csr(X)
        coarse_preds = self._coarse_model.predict(X)
        final_preds = np.empty(len(records), dtype=int)

        for i, coarse_id in enumerate(coarse_preds):
            coarse_id = int(coarse_id)
            if coarse_id not in self._limbo_models:
                raise ValueError(f"Coarse cluster {coarse_id} was not seen during fit")
            limbo, offset = self._limbo_models[coarse_id]
            label = limbo.predict([records[i]])[0]
            final_preds[i] = offset + label
        return final_preds

    def cluster_summary(self) -> Counter:
        if self.labels_ is None:
            raise RuntimeError("Call fit before requesting cluster summary")
        return Counter(int(x) for x in self.labels_.tolist())

    def coarse_summary(self, *, top_k: int = 5) -> Dict[str, Any]:
        if self._coarse_model is None:
            raise RuntimeError("Call fit before requesting coarse summary")
        return self._coarse_model.summary(top_k=top_k)

    @staticmethod
    def _ensure_csr(matrix: Any) -> Any:
        try:
            from scipy import sparse as sp  # type: ignore
        except Exception:  # pragma: no cover
            return matrix
        if sp.issparse(matrix):
            return matrix.tocsr(copy=True)
        return matrix


