"""Combine meaning-based search (dense embeddings) with keyword search (BM25).

The two scores live on different scales (cosine ~0.4-0.8, BM25 ~0-30), so we don't add them.
We combine RANKS instead: Reciprocal Rank Fusion (RRF)
    fused(chunk) = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)
A chunk ranked high by either method rises; ranked high by both rises most.
"""

import numpy as np

from bm25 import BM25
from embedding import embed
from retrieval import Retriever


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """rank[i] = 1-based position of chunk i when sorted by score (1 = best)."""
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks


class BM25Retriever:
    """Keyword-only search, same interface as Retriever (for evaluation)."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])

    def search(self, query: str, k: int = 3):
        scores = self.bm25.scores(query)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]


class HybridRetriever:
    def __init__(self, dense: Retriever, rrf_k: int = 60):
        self.dense = dense
        self.chunks = dense.chunks
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.rrf_k = rrf_k  # 60 is the value from the original RRF paper

    def search(self, query: str, k: int = 3):
        dense_scores = self.dense.vectors @ embed(self.dense.model, [query])[0]
        bm25_scores = self.bm25.scores(query)
        fused = 1 / (self.rrf_k + ranks_of(dense_scores)) + 1 / (self.rrf_k + ranks_of(bm25_scores))
        top = np.argsort(-fused)[:k]
        return [(float(fused[i]), self.chunks[i]) for i in top]
