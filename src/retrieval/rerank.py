"""Cross-encoder reranking: top-20 fused chunks -> keep top-5. (FR-11)

A cross-encoder reads the question and chunk TOGETHER as a pair and scores
how relevant the chunk is to the question. This is more accurate than dense
retrieval (which embeds question and chunk separately) but slower, so we only
run it on the top-20 fused candidates rather than the full corpus.

Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2 by default —
small, fast, runs on CPU, no API key needed.
"""
from __future__ import annotations

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(
    question: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Rerank chunks by cross-encoder score.

    chunks: list of dicts with keys chunk_id, text, metadata, fused_score
    Returns top_n chunks sorted by rerank score descending.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(question, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]