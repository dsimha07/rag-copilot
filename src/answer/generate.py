"""Generate a grounded answer then verify citations and score confidence."""
from __future__ import annotations
from src.common.llm import complete
from src.common.models import Answer, RetrievedChunk
from src.answer.prompt import SYSTEM, build_user_prompt
from src.answer.citations import verify_citations
from src.answer.confidence import compute_confidence


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    strategy: str = "hybrid",
    verify: bool = True,
) -> Answer:
    user_prompt = build_user_prompt(question, chunks)
    answer_text = complete(SYSTEM, user_prompt)

    if verify and chunks:
        citations, unverified = verify_citations(answer_text, chunks)
    else:
        citations, unverified = [], ""

    confidence = compute_confidence(
        answer_text=answer_text,
        question=question,
        citations=citations,
        retrieved_chunks=chunks,
    )

    return Answer(
        question=question,
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        unverified=unverified,
        retrieved_chunks=chunks,
        strategy=strategy,
    )