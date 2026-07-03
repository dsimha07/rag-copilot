"""Pydantic data contracts shared by both pipelines."""
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
    chunk_hash: str = ""
    embedding_version: str = ""


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

class Citation(BaseModel):
    chunk_id: str
    claim: str
    supported: bool
    evidence_span: str


class ConfidenceBreakdown(BaseModel):
    retrieval_score: float
    citation_support_rate: float
    completeness: float
    no_answer_flag: bool
    overall: float


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceBreakdown | None = None
    unverified: str = ""
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    strategy: str = "dense"