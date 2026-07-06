"""FastAPI /ask endpoint. (FR-20)

Exposes the full RAG pipeline over HTTP so any application can
call it programmatically — frontends, bots, test scripts, other services.

Endpoints:
  POST /ask      — ask a question, get a grounded cited answer
  GET  /health   — liveness check
  GET  /manifest — current index manifest (what's indexed)

Run with:
  uvicorn src.api.app:app --reload
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval.dense import retrieve as dense_retrieve
from src.retrieval.sparse import sparse_retrieve
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank
from src.answer.generate import answer_question
from src.common.models import RetrievedChunk, ChunkMetadata
from src.common.manifest import read_manifest
from src.ingest.index import get_collection

app = FastAPI(
    title="Support Knowledge Copilot",
    description="RAG pipeline with hybrid retrieval and verified citations.",
    version="1.0.0",
)


# ── request / response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    strategy: str = "hybrid"   # "hybrid" or "dense"
    verify: bool = True         # run citation verification


class CitationResponse(BaseModel):
    chunk_id: str
    claim: str
    supported: bool
    evidence_span: str


class ConfidenceResponse(BaseModel):
    overall: float
    retrieval_score: float
    citation_support_rate: float
    completeness: float
    no_answer_flag: bool


class SourceResponse(BaseModel):
    chunk_id: str
    source_name: str
    score: float | None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]
    confidence: ConfidenceResponse | None
    unverified: str
    sources: list[SourceResponse]
    strategy: str


# ── retrieval helper ──────────────────────────────────────────────────────────

def hybrid_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Full hybrid pipeline: dense + sparse → RRF → rerank → top_k."""
    dense_results = dense_retrieve(question, top_k=20)
    sparse_results = sparse_retrieve(question, top_k=20)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    col = get_collection()
    fused_chunks = []
    for chunk_id, fused_score in fused[:20]:
        res = col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            continue
        fused_chunks.append({
            "chunk_id": chunk_id,
            "text": res["documents"][0],
            "metadata": res["metadatas"][0],
            "fused_score": fused_score,
        })

    reranked = rerank(question, fused_chunks, top_n=top_k)
    out = []
    for c in reranked:
        out.append(RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            metadata=ChunkMetadata(**c["metadata"]),
            fused_score=c.get("fused_score"),
            rerank_score=c.get("rerank_score"),
        ))
    return out


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — confirms the API is running."""
    return {"status": "ok", "service": "rag-copilot"}


@app.get("/manifest")
def manifest():
    """Return the current index manifest — what's indexed and when."""
    m = read_manifest()
    if m is None:
        raise HTTPException(
            status_code=404,
            detail="No index manifest found. Run ingestion first."
        )
    return m.model_dump()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question and get a grounded cited answer.

    - strategy: "hybrid" (default) or "dense"
    - verify: True (default) runs citation verification via LLM judge
              False skips verification for faster responses
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.strategy not in ("hybrid", "dense"):
        raise HTTPException(status_code=400, detail="strategy must be 'hybrid' or 'dense'.")

    # retrieve
    if request.strategy == "dense":
        chunks = dense_retrieve(request.question)
    else:
        chunks = hybrid_retrieve(request.question)

    # generate + verify
    result = answer_question(
        question=request.question,
        chunks=chunks,
        strategy=request.strategy,
        verify=request.verify,
    )

    # build response
    citations = [
        CitationResponse(
            chunk_id=c.chunk_id,
            claim=c.claim,
            supported=c.supported,
            evidence_span=c.evidence_span,
        )
        for c in result.citations
    ]

    confidence = None
    if result.confidence:
        confidence = ConfidenceResponse(
            overall=result.confidence.overall,
            retrieval_score=result.confidence.retrieval_score,
            citation_support_rate=result.confidence.citation_support_rate,
            completeness=result.confidence.completeness,
            no_answer_flag=result.confidence.no_answer_flag,
        )

    sources = [
        SourceResponse(
            chunk_id=c.chunk_id,
            source_name=c.metadata.source_name,
            score=c.fused_score if c.fused_score is not None else c.dense_score,
        )
        for c in result.retrieved_chunks
    ]

    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=citations,
        confidence=confidence,
        unverified=result.unverified,
        sources=sources,
        strategy=result.strategy,
    )