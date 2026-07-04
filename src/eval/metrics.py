"""Retrieval and answer quality metrics. (FR-18)

Retrieval and answer metrics are kept SEPARATE deliberately.
"""
from __future__ import annotations
from src.eval.golden import GoldenEntry
from src.common.models import Answer


def correct_source_retrieved(entry: GoldenEntry, answer: Answer) -> bool:
    if not entry.expected_chunks:
        return True
    retrieved_ids = {c.chunk_id for c in answer.retrieved_chunks}
    return bool(retrieved_ids & set(entry.expected_chunks))


def reciprocal_rank(entry: GoldenEntry, answer: Answer) -> float:
    if not entry.expected_chunks:
        return 1.0
    retrieved_ids = [c.chunk_id for c in answer.retrieved_chunks]
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in entry.expected_chunks:
            return 1.0 / rank
    return 0.0


def answer_correct(entry: GoldenEntry, answer: Answer) -> bool:
    lower = answer.answer.lower()
    return all(exp.lower() in lower for exp in entry.expected_answer_contains)


def correct_refusal(entry: GoldenEntry, answer: Answer) -> bool | None:
    if not entry.no_answer:
        return None
    if answer.confidence and answer.confidence.no_answer_flag:
        return True
    lower = answer.answer.lower()
    refusal_phrases = [
        "could not find", "not found in", "not in the documentation",
        "not mentioned", "no information", "cannot find", "not available",
        "unable to find",
    ]
    return any(phrase in lower for phrase in refusal_phrases)


def citation_support_rate(answer: Answer) -> float:
    if not answer.citations:
        return 0.5
    supported = sum(1 for c in answer.citations if c.supported)
    return round(supported / len(answer.citations), 4)


def score_entry(entry: GoldenEntry, answer: Answer) -> dict:
    return {
        "id": entry.id,
        "category": entry.category,
        "difficulty": entry.difficulty,
        "question": entry.question,
        "correct_source": correct_source_retrieved(entry, answer),
        "reciprocal_rank": reciprocal_rank(entry, answer),
        "answer_correct": answer_correct(entry, answer),
        "correct_refusal": correct_refusal(entry, answer),
        "citation_support_rate": citation_support_rate(answer),
        "confidence": answer.confidence.overall if answer.confidence else None,
        "answer_text": answer.answer[:200],
        "retrieved_chunks": [c.chunk_id for c in answer.retrieved_chunks],
    }


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    answerable = [r for r in results if r["correct_refusal"] is None]
    no_answer  = [r for r in results if r["correct_refusal"] is not None]

    correct_sources = sum(1 for r in results if r["correct_source"])
    mrr = sum(r["reciprocal_rank"] for r in results) / total if total else 0
    correct_answers = sum(1 for r in answerable if r["answer_correct"])
    correct_refusals = sum(1 for r in no_answer if r["correct_refusal"])
    avg_csr = sum(r["citation_support_rate"] for r in results) / total if total else 0
    confidences = [r["confidence"] for r in results if r["confidence"] is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if r["answer_correct"] or r["correct_refusal"]:
            by_category[cat]["correct"] += 1

    return {
        "total": total,
        "retrieval": {
            "correct_source_retrieved": correct_sources,
            "correct_source_pct": round(correct_sources / total * 100, 1) if total else 0,
            "mean_reciprocal_rank": round(mrr, 4),
        },
        "answer": {
            "correct_answers": correct_answers,
            "total_answerable": len(answerable),
            "correct_answer_pct": round(correct_answers / len(answerable) * 100, 1) if answerable else 0,
            "correct_refusals": correct_refusals,
            "total_no_answer": len(no_answer),
            "correct_refusal_pct": round(correct_refusals / len(no_answer) * 100, 1) if no_answer else 0,
            "avg_citation_support_rate": round(avg_csr, 4),
            "avg_confidence": round(avg_conf, 4),
        },
        "by_category": by_category,
    }