"""BM25 sparse retrieval over the same chunk IDs as the dense index. (FR-9)

Builds a BM25 index from the corpus chunks and scores a query against it.
Keyword search catches exact matches — error codes, emails, product names,
specific phrases — that semantic/dense search misses.

Index is saved to disk as a pickle file so it persists between runs, the same
way ChromaDB persists the dense index.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.common import config
from src.common.models import Chunk

BM25_PATH = config.ROOT / "data" / "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list[Chunk]) -> None:
    """Build a BM25 index from chunks and save it to disk."""
    corpus = [_tokenize(c.text) for c in chunks]
    chunk_ids = [c.metadata.chunk_id for c in chunks]
    index = BM25Okapi(corpus)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": index, "chunk_ids": chunk_ids}, f)


def load_bm25_index() -> tuple[BM25Okapi, list[str]]:
    """Load the BM25 index from disk."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. Run ingestion first."
        )
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunk_ids"]


def sparse_retrieve(question: str, top_k: int | None = None) -> list[tuple[str, float]]:
    """Return (chunk_id, bm25_score) pairs ranked by BM25 score."""
    top_k = top_k or config.TOP_K
    index, chunk_ids = load_bm25_index()
    scores = index.get_scores(_tokenize(question))
    ranked = sorted(
        zip(chunk_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]