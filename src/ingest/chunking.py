"""Fixed-size chunking with overlap. (FR-5 — one strategy for the skeleton;
recursive heading-based is added during hardening, tagged so the two can be
compared.)
"""
from __future__ import annotations
import hashlib
from src.common.models import Chunk, ChunkMetadata
from src.common import config


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(doc: dict, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = doc["text"]
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            meta = ChunkMetadata(
                chunk_id=f"{doc['doc_id']}::{idx}",
                doc_id=doc["doc_id"],
                source_name=doc["source_name"],
                source_path=doc["source_path"],
                chunking_strategy="fixed_overlap",
                chunk_index=idx,
                chunk_hash=_hash(piece),
                embedding_version=config.EMBEDDING_MODEL,
            )
            chunks.append(Chunk(text=piece, metadata=meta))
            idx += 1
        start += step
    return chunks
