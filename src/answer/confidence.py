"""Confidence scoring with breakdown. (FR-14)"""
from __future__ import annotations
from src.common.models import Citation, ConfidenceBreakdown, RetrievedChunk

WEIGHTS = {
    "retrieval": 0.35,
    "citation_support": 0.40,
    "completeness": 0.25,
}

NO_ANSWER_PHRASES = [
    "could not find", "not found in", "not in the documentation",
    "not mentioned in", "no information", "cannot find",
    "don't have information", "not covered", "not available in",
    "unable to find",
]


def _retrieval_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
     # prefer fused_score (hybrid path), fall back to dense_score
    scores = [
        c.fused_score if c.fused_score is not None else c.dense_score
        for c in chunks
        if (c.fused_score is not None or c.dense_score is not None)
    ]
    if not scores:
        return 0.5
    avg = sum(scores) / len(scores)
    scaled = min(avg / 0.0164, 1.0)
    return round(max(0.0, scaled), 4)


def _citation_support_rate(citations: list[Citation]) -> float:
    if not citations:
        return 0.5
    supported = sum(1 for c in citations if c.supported)
    return round(supported / len(citations), 4)


def _completeness_score(answer_text: str, question: str) -> float:
    if not answer_text or len(answer_text.strip()) < 30:
        return 0.1
    lower = answer_text.lower()
    for phrase in NO_ANSWER_PHRASES:
        if phrase in lower:
            return 0.3
    length_score = min(len(answer_text) / 300, 1.0)
    return round(length_score, 4)


def _no_answer_flag(answer_text: str) -> bool:
    lower = answer_text.lower()
    return any(phrase in lower for phrase in NO_ANSWER_PHRASES)


def compute_confidence(
    answer_text: str,
    question: str,
    citations: list[Citation],
    retrieved_chunks: list[RetrievedChunk],
) -> ConfidenceBreakdown:
    retrieval = _retrieval_score(retrieved_chunks)
    citation_support = _citation_support_rate(citations)
    completeness = _completeness_score(answer_text, question)
    no_answer = _no_answer_flag(answer_text)

    overall = (
        WEIGHTS["retrieval"] * retrieval
        + WEIGHTS["citation_support"] * citation_support
        + WEIGHTS["completeness"] * completeness
    )
    if no_answer:
        overall = min(overall, 0.4)

    return ConfidenceBreakdown(
        retrieval_score=retrieval,
        citation_support_rate=citation_support,
        completeness=completeness,
        no_answer_flag=no_answer,
        overall=round(overall, 4),
    )