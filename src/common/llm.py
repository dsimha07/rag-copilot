"""Anthropic (Claude) chat wrapper. (NFR-4)

Generation runs on Claude Sonnet. Note this is independent from embeddings,
which stay on OpenAI — Anthropic does not provide an embeddings endpoint.

Anthropic SDK differences vs OpenAI to keep in mind:
  - the system prompt is a top-level `system=` argument, not a message
  - `max_tokens` is required
  - the reply is a list of content blocks; text is at content[0].text
"""
from __future__ import annotations
from src.common import config

_client = None


def _get_client():
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def complete(system: str, user: str, model: str | None = None, max_tokens: int = 1024) -> str:
    resp = _get_client().messages.create(
        model=model or config.LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    return resp.content[0].text if resp.content else ""