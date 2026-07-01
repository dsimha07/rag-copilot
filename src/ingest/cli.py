"""Ingest CLI:  python -m src.ingest.cli --rebuild   (FR-7)"""
from __future__ import annotations
import argparse
from pathlib import Path
from src.common import config
from src.ingest.loaders import load_markdown
from src.ingest.chunking import chunk_document
from src.ingest.index import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the support corpus.")
    parser.add_argument("--source", default=str(config.CORPUS_DIR))
    parser.add_argument("--rebuild", action="store_true", help="reset the index first")
    args = parser.parse_args()

    docs = load_markdown(Path(args.source))
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    n = index_chunks(all_chunks, reset=args.rebuild)
    print(f"Indexed {n} chunks from {len(docs)} documents into '{config.COLLECTION}'.")


if __name__ == "__main__":
    main()
