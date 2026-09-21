"""Text embedding with an offline-by-default backend.

The default ``hashing`` backend is a signed hashing-trick projection of word
unigrams, bigrams and character 4-grams - effectively a random projection of a
TF-IDF vector. It is deterministic, needs no downloads, and runs in
microseconds, which matters because retrieval has to work on a laptop with no
network and because a counterfactual re-run must retrieve *the same*
neighbours as the original prediction.

It captures lexical and morphological overlap rather than deep semantics. When
richer similarity is wanted, set ``EMBEDDING_BACKEND=sentence-transformers``
and the same interface loads a local transformer instead. Retrieval never
relies on text similarity alone: :mod:`app.services.embeddings.store` blends it
with factor-space similarity, which is the part that carries decision meaning.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "are", "were", "be", "been", "it", "this", "that", "i",
    "my", "me", "we", "you", "as", "by", "from", "so", "than", "then", "there",
}


class Embedder(ABC):
    """Common interface for every embedding backend."""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _features(text: str) -> list[str]:
    """Word unigrams, bigrams and character 4-grams of longer words."""
    tokens = _tokens(text)
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    features: list[str] = list(content)
    features.extend(f"{a}_{b}" for a, b in zip(content, content[1:], strict=False))
    for token in content:
        if len(token) >= 6:
            features.extend(f"#{token[i:i + 4]}" for i in range(len(token) - 3))
    return features


class HashingEmbedder(Embedder):
    """Signed hashing trick with sublinear term weighting."""

    name = "hashing"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim or settings.embedding_dim)

    @staticmethod
    def _hash(feature: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        features = _features(text)
        if not features:
            return vector

        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        for feature, count in counts.items():
            digest = self._hash(feature)
            bucket = digest % self.dim
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            # Sublinear tf: the fifth mention of "salary" adds little.
            weight = 1.0 + math.log(count)
            # Character n-grams are noisier than whole words.
            if feature.startswith("#"):
                weight *= 0.45
            elif "_" in feature:
                weight *= 0.8
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover - optional path
    """Local transformer embeddings, when the package is installed."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or settings.embedding_model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text or "", normalize_embeddings=True)
        return [float(v) for v in vector]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(v) for v in row] for row in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Build the configured embedder, falling back loudly rather than silently."""
    backend = (settings.embedding_backend or "hashing").lower()
    if backend in {"sentence-transformers", "st", "transformer"}:
        try:
            embedder = SentenceTransformerEmbedder()
            log.info("using sentence-transformers embeddings (dim=%d)", embedder.dim)
            return embedder
        except Exception as exc:
            log.warning(
                "sentence-transformers unavailable (%s); falling back to hashing embeddings", exc
            )
    return HashingEmbedder()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, safe on zero vectors and mismatched lengths."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = na = nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / math.sqrt(na * nb))
