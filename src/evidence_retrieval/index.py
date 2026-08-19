"""Small, deterministic dense index used by smoke experiments and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    passage_id: str
    score: float


class ExactDotProductIndex:
    """An inspectable in-memory index; use FAISS only after this baseline is verified."""

    def __init__(self, passage_ids: Sequence[str], embeddings: np.ndarray) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must have shape [count, dimension]")
        if len(passage_ids) != vectors.shape[0]:
            raise ValueError("passage_ids and embeddings must have matching lengths")
        if len(set(passage_ids)) != len(passage_ids):
            raise ValueError("passage_ids must be unique")
        self.passage_ids = list(passage_ids)
        self.embeddings = vectors

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query_embedding has incompatible shape")
        scores = self.embeddings @ query
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchResult(self.passage_ids[index], float(scores[index])) for index in indices]
