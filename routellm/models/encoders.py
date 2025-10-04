from __future__ import annotations

from typing import List, Optional

import hashlib
import numpy as np


class BaseTextEncoder:
    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim: int = embedding_dim

    def encode(self, texts: List[str], show_progress_bar: Optional[bool] = None) -> np.ndarray:
        raise NotImplementedError


class HashFallbackEncoder(BaseTextEncoder):
    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__(embedding_dim)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        if not text:
            return vector
        for n in (3, 4, 5):
            for i in range(0, max(1, len(text) - n + 1)):
                ngram = text[i : i + n]
                h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                idx = h % self.embedding_dim
                value = ((h >> 8) % 1000) / 1000.0
                vector[idx] += value
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def encode(self, texts: List[str], show_progress_bar: Optional[bool] = None) -> np.ndarray:
        return np.vstack([self._encode_one(t) for t in texts])


class SentenceTransformerEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384) -> None:
        super().__init__(embedding_dim)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers 未安装") from exc
        self._model = SentenceTransformer(model_name)
        self.embedding_dim = getattr(self._model, "get_sentence_embedding_dimension", lambda: embedding_dim)()

    def encode(self, texts: List[str], show_progress_bar: Optional[bool] = None) -> np.ndarray:
        # 仅在批量较大时显示内部进度条，避免小批量刷屏
        if show_progress_bar is None:
            try:
                show_progress_bar = len(texts) >= 32
            except Exception:
                show_progress_bar = False
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=bool(show_progress_bar))
        return np.asarray(embeddings, dtype=np.float32)


def build_text_encoder(prefer_transformer: bool = True, embedding_dim: int = 384) -> BaseTextEncoder:
    if prefer_transformer:
        try:
            return SentenceTransformerEncoder()
        except Exception:
            return HashFallbackEncoder(embedding_dim=embedding_dim)
    return HashFallbackEncoder(embedding_dim=embedding_dim)


class CategoricalFeatureVectorizer:
    def __init__(self) -> None:
        self._feature_to_index: dict[str, dict[str, int]] = {}
        self._dim: int = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, feature_dicts: List[dict[str, str]]) -> None:
        value_spaces: dict[str, set[str]] = {}
        for item in feature_dicts:
            for key, value in (item or {}).items():
                value_spaces.setdefault(key, set()).add(str(value))
        offset = 0
        self._feature_to_index = {}
        for key, values in sorted(value_spaces.items()):
            mapping: dict[str, int] = {}
            for i, v in enumerate(sorted(values)):
                mapping[v] = offset + i
            self._feature_to_index[key] = mapping
            offset += len(mapping)
        self._dim = offset

    def transform(self, feature_dicts: List[Optional[dict[str, str]]]) -> np.ndarray:
        if self._dim == 0:
            return np.zeros((len(feature_dicts), 0), dtype=np.float32)
        mat = np.zeros((len(feature_dicts), self._dim), dtype=np.float32)
        for row, item in enumerate(feature_dicts):
            if not item:
                continue
            for key, value in item.items():
                mapping = self._feature_to_index.get(key)
                if not mapping:
                    continue
                idx = mapping.get(str(value))
                if idx is None:
                    continue
                mat[row, idx] = 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

