"""Config + paths for the walking skeleton. (NFR-1 — minimal for now.)"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "support_docs"

# Embeddings stay on OpenAI — Anthropic has no embeddings endpoint.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Generation is Claude Sonnet.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # embeddings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # generation

# fixed-size chunking (characters) — one strategy for the skeleton
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
