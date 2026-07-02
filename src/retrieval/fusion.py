"""Reciprocal Rank Fusion (RRF) of dense and sparse result lists. (FR-10)

RRF merges two ranked lists into one without needing comparable scores.
It only cares about rank position, not the actual score values — which is
exactly what you need when combining cosine similarity (dense) with BM25
scores (sparse) since those two numbers live on completely different scales.

Formula per chunk: score = sum of 1 / (k + rank) across all lists
where k=60 is a constant that dampens the impact of very high ranks.
Higher fused score = more relevant.
"""
from __future__ import annotations
from src.common.models import RetrievedChunk

K = 60


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[tuple[str, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Fuse dense and sparse results using weighted RRF.

    Returns a list of (chunk_id, fused_score) sorted by fused_score descending.
    Weights let you control how much each retrieval method contributes.
    """
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (K + rank + 1))

    for rank, (cid, _) in enumerate(sparse_results):
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (K + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)