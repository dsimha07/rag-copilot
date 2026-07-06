"""Build the Chroma index AND the BM25 index over the SAME chunk IDs. (FR-6)

Both indexes must always be built together and kept in sync — they are two
views of the same chunks. Chroma stores dense vectors for semantic search;
the BM25 pickle stores tokenized text for keyword search.
"""
from __future__ import annotations
import chromadb
from src.common import config
from src.common.embeddings import embed_texts
from src.common.models import Chunk


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(config.COLLECTION)


def _clean_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if v is not None}


def index_chunks(chunks: list[Chunk],docs: list[dict] | None = None, reset: bool = False) -> int:
    """Embed chunks into Chroma AND build the BM25 index."""
    col = get_collection(reset=reset)
    if not chunks:
        return 0

    # --- dense index (Chroma) ---
    embeddings = embed_texts([c.text for c in chunks])
    col.add(
        ids=[c.metadata.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_clean_meta(c.metadata.model_dump()) for c in chunks],
    )

    # --- sparse index (BM25) ---
    from src.retrieval.sparse import build_bm25_index
    build_bm25_index(chunks)

    # manifest
    if docs is not None:
        from src.common.manifest import build_manifest, write_manifest
        chunks_per_doc: dict[str, int] = {}
        for c in chunks:
            chunks_per_doc[c.metadata.doc_id] = chunks_per_doc.get(c.metadata.doc_id, 0) + 1
        manifest = build_manifest(docs, chunks_per_doc)
        write_manifest(manifest)

    return len(chunks)