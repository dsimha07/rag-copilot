"""Citation parsing and verification. (FR-13, FR-16)"""
from __future__ import annotations
import re
from src.common.judge import judge_citation
from src.common.models import Citation, RetrievedChunk

CITATION_PATTERN = re.compile(r'\[([a-zA-Z0-9_]+::[0-9]+)\]')


def _clean(text: str) -> str:
    text = CITATION_PATTERN.sub('', text)
    text = re.sub(r'[*#`_]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_citations(answer_text: str) -> list[tuple[str, str]]:
    pairs = []
    lines = answer_text.splitlines()
    last_claim = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        tags = CITATION_PATTERN.findall(stripped)
        text_part = _clean(stripped)

        if text_part and len(text_part) >= 5:
            last_claim = text_part
            for tag in tags:
                pairs.append((last_claim, tag))
        elif tags and not text_part:
            if last_claim:
                for tag in tags:
                    pairs.append((last_claim, tag))

    return pairs


def verify_citations(
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[list[Citation], str]:
    chunk_lookup = {c.chunk_id: c.text for c in retrieved_chunks}
    pairs = parse_citations(answer_text)
    citations: list[Citation] = []
    failed: list[str] = []

    for claim, chunk_id in pairs:
        chunk_text = chunk_lookup.get(chunk_id)

        if chunk_text is None:
            citations.append(Citation(
                chunk_id=chunk_id, claim=claim,
                supported=False, evidence_span="",
            ))
            failed.append(
                f'"{claim}" — cited [{chunk_id}] but chunk was not retrieved'
            )
            continue

        verdict = judge_citation(claim, chunk_text)
        supported = verdict.get("supported", False)
        evidence = verdict.get("evidence", "")

        citations.append(Citation(
            chunk_id=chunk_id, claim=claim,
            supported=supported, evidence_span=evidence,
        ))

        if not supported:
            reason = verdict.get("reason", "no reason provided")
            failed.append(f'"{claim}" — cited [{chunk_id}] but {reason}')

    unverified = "\n".join(f"- {f}" for f in failed) if failed else ""
    return citations, unverified