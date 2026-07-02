"""End-to-end runner: ask one question through the full hybrid pipeline.

Pipeline:
  question
    → dense retrieval (top 20)
    → sparse BM25 retrieval (top 20)
    → RRF fusion
    → cross-encoder rerank (top 20 fused → top 5)
    → Claude Sonnet generation
    → print answer + sources

Usage:
    python ask.py "How do I reset my password?"
    python ask.py --strategy dense "How do I reset my password?"
"""
from __future__ import annotations
import sys
import argparse
from src.retrieval.dense import retrieve as dense_retrieve
from src.retrieval.sparse import sparse_retrieve
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank
from src.answer.generate import answer_question
from src.common.models import RetrievedChunk, ChunkMetadata
from src.ingest.index import get_collection


def hybrid_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Full hybrid pipeline: dense + sparse → RRF → rerank → top_k."""

    # step 1 — dense top 20
    dense_results = dense_retrieve(question, top_k=20)

    # step 2 — sparse top 20
    sparse_results = sparse_retrieve(question, top_k=20)

    # step 3 — RRF fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    # step 4 — fetch full chunk text for fused results
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

    # step 5 — cross-encoder rerank → top 5
    reranked = rerank(question, fused_chunks, top_n=top_k)

    # step 6 — convert to RetrievedChunk objects
    out = []
    for c in reranked:
        out.append(RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            metadata=ChunkMetadata(**c["metadata"]),
            dense_score=c.get("fused_score"),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--strategy",
        choices=["dense", "hybrid"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid)"
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    if args.strategy == "dense":
        from src.retrieval.dense import retrieve
        chunks = retrieve(question)
    else:
        chunks = hybrid_retrieve(question)

    result = answer_question(question, chunks)

    print(f"\n=== Answer ({args.strategy}) ===")
    print(result.answer)
    print("\n=== Sources ===")
    for c in result.retrieved_chunks:
        score = f"{c.dense_score:.4f}" if c.dense_score is not None else "n/a"
        print(f"  [{c.chunk_id}] {c.metadata.source_name}  (score {score})")


if __name__ == "__main__":
    main()