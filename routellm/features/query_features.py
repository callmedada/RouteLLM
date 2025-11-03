from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence


class SpaCyNotInstalledError(RuntimeError):
    """Raised when spaCy or the requested model is missing."""


def _require_spacy(spacy_model: str, enable_components: Sequence[str]):
    try:
        import spacy  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SpaCyNotInstalledError(
            "spaCy 未安装，请先运行 'pip install spacy'"
        ) from exc

    try:
        disabled = [
            name
            for name in ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter")
            if name not in enable_components
        ]
        nlp = spacy.load(spacy_model, disable=disabled)
    except OSError as exc:  # pragma: no cover - runtime guard
        raise SpaCyNotInstalledError(
            f" spaCy nned '{spacy_model}'， 'python -m spacy download {spacy_model}'"
        ) from exc
    return nlp


@dataclass(slots=True)
class QueryFeatureExtractor:
    hash_buckets: int = 128
    spacy_model: str = "en_core_web_sm"
    spacy_components: Sequence[str] = ("tok2vec", "tagger", "parser", "ner")
    pos_top_k: int = 3
    hash_ngrams: Sequence[int] = (3, 4, 5)
    max_hash_per_ngram: int = 64
    punct_chars: str = "!?,.;:"
    url_pattern: re.Pattern[str] = re.compile(r"https?://|www\.", re.IGNORECASE)
    math_pattern: re.Pattern[str] = re.compile(r"[=<>±×÷∑√∫π∞+\-*/]")
    code_pattern: re.Pattern[str] = re.compile(r"```|\bdef\b|\bclass\b")
    _nlp: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nlp = _require_spacy(self.spacy_model, self.spacy_components)
        # 预热：如果缺失 required pipeline，会即时报错
        required = set(self.spacy_components)
        if "parser" in required or "senter" in required:
            if "senter" not in self._nlp.pipe_names and "parser" not in self._nlp.pipe_names:
                raise SpaCyNotInstalledError(
                    "spaCy pipeline 缺少 parser/senter，。"
                )

    def extract(self, texts: Sequence[str]) -> List[dict[str, str]]:
        docs = list(self._nlp.pipe(texts, batch_size=32))
        out: List[dict[str, str]] = []
        for text, doc in zip(texts, docs):
            feats: dict[str, str] = {}
            feats.update(self._basic_features(text))
            feats.update(self._hash_features(text))
            feats.update(self._spacy_features(doc))
            out.append(feats)
        return out

    # -------------------------
    # basic textual heuristics
    # -------------------------
    def _basic_features(self, text: str) -> dict[str, str]:
        length = len(text)
        tokens = text.split()
        token_count = len(tokens)
        feats: dict[str, str] = {}
        feats["len_bucket"] = self._bucket(length, [32, 64, 128, 256, 512, 1024])
        feats["token_bucket"] = self._bucket(token_count, [8, 16, 32, 64, 128])
        number_ratio = self._safe_ratio(sum(ch.isdigit() for ch in text), max(1, length))
        feats["has_number"] = "1" if number_ratio > 0 else "0"
        feats["number_density_bucket"] = self._bucket(int(number_ratio * 100), [1, 5, 10, 20])
        math_ratio = self._safe_ratio(len(self.math_pattern.findall(text)), max(1, token_count))
        feats["has_math_symbol"] = "1" if math_ratio > 0 else "0"
        feats["math_density_bucket"] = self._bucket(int(math_ratio * 100), [1, 5, 10])
        code_ratio = self._safe_ratio(len(self.code_pattern.findall(text)), max(1, token_count))
        feats["has_code_pattern"] = "1" if code_ratio > 0 else "0"
        punct_ratio = self._safe_ratio(sum(ch in self.punct_chars for ch in text), max(1, token_count))
        feats["punct_density_bucket"] = self._bucket(int(punct_ratio * 100), [10, 20, 40, 60])
        feats["contains_url"] = "1" if self.url_pattern.search(text) else "0"
        upper_ratio = self._safe_ratio(sum(word.isupper() for word in tokens if word.isalpha()), max(1, token_count))
        feats["upper_density_bucket"] = self._bucket(int(upper_ratio * 100), [5, 15, 30, 60])
        return feats

    def _hash_features(self, text: str) -> dict[str, str]:
        text = text.lower().strip()
        feats: dict[str, str] = {}
        for n in self.hash_ngrams:
            seen: set[int] = set()
            limit = 0
            for i in range(0, max(0, len(text) - n + 1)):
                ngram = text[i : i + n]
                idx = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % max(1, self.hash_buckets)
                if idx in seen:
                    continue
                seen.add(idx)
                feats[f"hash{n}_{idx}"] = "1"
                limit += 1
                if limit >= self.max_hash_per_ngram:
                    break
        return feats

    # -------------------------
    # spaCy-driven features
    # -------------------------
    def _spacy_features(self, doc) -> dict[str, str]:  # type: ignore[no-untyped-def]
        feats: dict[str, str] = {}
        # POS top-k
        pos_counts = Counter(token.pos_ for token in doc)
        for rank, (pos_tag, _) in enumerate(pos_counts.most_common(self.pos_top_k), start=1):
            feats[f"pos_top{rank}"] = (pos_tag or "UNK").upper()

        # NER flags
        if doc.ents:
            ent_labels = {ent.label_.upper() for ent in doc.ents}
            for label in ent_labels:
                feats[f"has_ent_{label}"] = "1"
        else:
            feats["has_ent_NONE"] = "1"

        # sentence/token stats
        sent_count = sum(1 for _ in doc.sents)
        feats["sent_bucket"] = self._bucket(sent_count, [1, 2, 3, 5, 8])
        feats["spacy_token_bucket"] = self._bucket(len(doc), [16, 32, 64, 128, 256])

        # shape / orthographic features
        if len(doc) > 0:
            shape_counts = Counter(token.shape_ for token in doc if token.shape_)
            shape_ranked = shape_counts.most_common(2)
            for rank, (shape, _) in enumerate(shape_ranked, start=1):
                feats[f"shape_top{rank}"] = shape[:12]
            title_ratio = self._safe_ratio(sum(1 for token in doc if token.is_title), len(doc))
            feats["title_density_bucket"] = self._bucket(int(title_ratio * 100), [5, 15, 30, 50])
        return feats

    # -------------------------
    # helper utilities
    # -------------------------
    @staticmethod
    def _bucket(value: int, thresholds: Iterable[int]) -> str:
        thrs = list(thresholds)
        for idx, thr in enumerate(thrs):
            if value <= thr:
                return f"{idx}"
        return f"{len(thrs)}"

    @staticmethod
    def _safe_ratio(numer: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return float(numer) / float(denom)


