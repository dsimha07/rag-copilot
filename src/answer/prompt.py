"""Grounded-answer prompt for the skeleton. Strict citation enforcement and a
no-answer contract are tightened during hardening. (FR-12)
"""
from __future__ import annotations
from src.common.models import RetrievedChunk

SYSTEM = (
    "You are a support knowledge assistant. "
    "Answer ONLY using the provided context. "
    "After EACH sentence that uses information from the context, "
    "immediately cite the source chunk ID inline like this: [faq::0] "
    "Place the citation right after the sentence it supports, not at the end. "
    "If the answer is not in the context, say you could not find it in the documentation. "
    "Be concise."
)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{c.chunk_id}] (source: {c.metadata.source_name})\n{c.text}"
        for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
