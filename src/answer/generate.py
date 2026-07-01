"""Generate a grounded answer from retrieved chunks. (FR-12)

Walking skeleton: returns answer text + the chunks used. Citation parsing,
verification, and confidence scoring are added during hardening.
"""
from __future__ import annotations
from src.common.llm import complete
from src.common.models import Answer, RetrievedChunk
from src.answer.prompt import SYSTEM, build_user_prompt


def answer_question(question: str, chunks: list[RetrievedChunk]) -> Answer:
    user = build_user_prompt(question, chunks)
    text = complete(SYSTEM, user)
    return Answer(question=question, answer=text, retrieved_chunks=chunks, strategy="dense")
