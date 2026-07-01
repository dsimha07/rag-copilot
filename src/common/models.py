"""Minimal data contracts for the walking skeleton.

Only the shapes the dense-retrieval slice needs. The full contract (Citation,
ConfidenceBreakdown, IndexManifest) is added during hardening — see
REQUIREMENTS_project1.md section 4. chunk_hash + embedding_version are kept now
so Project 7 needs no retrofit.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    source_name: str
    source_path: str
    section_heading: str | None = None
    doc_type: str = "unknown"
    last_modified: str | None = None
    access_level: str = "internal"
    chunking_strategy: str = "fixed_overlap"
    chunk_index: int = 0
    chunk_hash: str = ""          # sha256 of cleaned chunk text (used by Project 7)
    embedding_version: str = ""   # embedding model id (used by Project 7)


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_score: float | None = None


class Answer(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    strategy: str = "dense"
