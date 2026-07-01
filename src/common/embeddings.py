"""OpenAI embeddings. The SAME function is used at ingestion AND query time so
the vector spaces match — non-negotiable. (NFR-4)
"""
from __future__ import annotations
from src.common import config

EMBEDDING_VERSION = config.EMBEDDING_MODEL
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = _get_client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
