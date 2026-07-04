"""Eval CLI: python -m src.eval.run --strategy hybrid (FR-19)

run.py checks whether the RAG pipeline gives the right answers to questions you already know the answers to.

Runs every question in the golden set through the pipeline,
scores each one, aggregates results, and writes a markdown report.

Usage:
    python -m src.eval.run                      # hybrid, with verification
    python -m src.eval.run --strategy dense     # dense only
    python -m src.eval.run --no-verify          # skip citation verification (faster)
    python -m src.eval.run --limit 5            # only run first 5 questions
"""
from __future__ import annotations
import argparse
from src.eval.golden import load_golden
from src.eval.metrics import score_entry, aggregate
from src.eval.report import generate_report
from src.answer.generate import answer_question
from src.retrieval.dense import retrieve as dense_retrieve
from src.ingest.index import get_collection
from src.retrieval.sparse import sparse_retrieve
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank
from src.common.models import RetrievedChunk, ChunkMetadata


def hybrid_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Same hybrid pipeline as ask.py."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden Q&A eval set.")
    parser.add_argument(
        "--strategy", choices=["dense", "hybrid"], default="hybrid"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip citation verification (faster)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (for quick testing)"
    )
    args = parser.parse_args()

    entries = load_golden()
    if args.limit:
        entries = entries[:args.limit]

    print(f"Running {len(entries)} questions — strategy={args.strategy} verify={not args.no_verify}")
    print("-" * 60)

    results = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i:02d}/{len(entries)}] {entry.id} — {entry.question[:50]}...")

        # retrieve
        if args.strategy == "dense":
            chunks = dense_retrieve(entry.question)
        else:
            chunks = hybrid_retrieve(entry.question)

        # generate + optionally verify
        answer = answer_question(
            question=entry.question,
            chunks=chunks,
            strategy=args.strategy,
            verify=not args.no_verify,
        )

        # score
        result = score_entry(entry, answer)
        results.append(result)

        source_ok = "✅" if result["correct_source"] else "❌"
        answer_ok = "✅" if result["answer_correct"] else "❌"
        refusal = f" refusal={'✅' if result['correct_refusal'] else '❌'}" if entry.no_answer else ""
        conf = f"{result['confidence']:.2f}" if result["confidence"] else "n/a"
        print(f"         source={source_ok} answer={answer_ok}{refusal} conf={conf}")

    print("-" * 60)
    aggregated = aggregate(results)
    ret = aggregated["retrieval"]
    ans = aggregated["answer"]
    print(f"correct source retrieved: {ret['correct_source_retrieved']}/{aggregated['total']} ({ret['correct_source_pct']}%)")
    print(f"mean reciprocal rank:     {ret['mean_reciprocal_rank']}")
    print(f"answers correct:          {ans['correct_answers']}/{ans['total_answerable']} ({ans['correct_answer_pct']}%)")
    print(f"correct refusals:         {ans['correct_refusals']}/{ans['total_no_answer']} ({ans['correct_refusal_pct']}%)")
    print(f"avg citation support:     {ans['avg_citation_support_rate']}")
    print(f"avg confidence:           {ans['avg_confidence']}")

    report_path = generate_report(results, aggregated, args.strategy)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()