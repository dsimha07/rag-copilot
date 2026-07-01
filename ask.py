"""Walking-skeleton runner: ask one question end to end.

Usage:
    python ask.py "How do I reset my password?"
"""
from __future__ import annotations
import sys
from src.retrieval.dense import retrieve
from src.answer.generate import answer_question


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    chunks = retrieve(question)
    result = answer_question(question, chunks)

    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Sources ===")
    for c in result.retrieved_chunks:
        score = f"{c.dense_score:.3f}" if c.dense_score is not None else "n/a"
        print(f"  [{c.chunk_id}] {c.metadata.source_name}  (score {score})")


if __name__ == "__main__":
    main()
