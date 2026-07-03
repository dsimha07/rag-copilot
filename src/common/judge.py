"""LLM-as-judge harness.

Used by Project 1 citation verification AND (later) Project 7 answer drift.
Takes a claim and a chunk, asks Claude Sonnet whether the chunk supports
the claim, and returns a structured verdict with evidence.
"""
from __future__ import annotations
import json
from src.common.llm import complete

JUDGE_SYSTEM = """You are a precise fact-checking judge.

You will be given:
- A CLAIM: one sentence from an AI-generated answer
- A CHUNK: the source text the claim was supposedly drawn from

Your job is to determine whether the CHUNK actually supports the CLAIM.

Rules:
- Only use what is explicitly stated in the CHUNK
- Do not use outside knowledge
- Be strict — if the chunk is vague or doesn't directly support the claim, mark it unsupported

Respond with ONLY a JSON object in this exact format:
{
  "supported": true or false,
  "evidence": "the exact phrase from the chunk that supports the claim, or empty string if unsupported",
  "reason": "one sentence explaining your verdict"
}"""


def judge_citation(claim: str, chunk_text: str) -> dict:
    """
    Ask the judge whether chunk_text supports claim.

    Returns:
        {
          "supported": bool,
          "evidence": str,
          "reason": str
        }
    """
    user = f"CLAIM: {claim}\n\nCHUNK: {chunk_text}"
    raw = complete(JUDGE_SYSTEM, user, max_tokens=256)

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return {
            "supported": False,
            "evidence": "",
            "reason": f"judge response could not be parsed: {raw[:100]}"
        }