"""Build the Chroma index: embed chunks and store with metadata. (FR-6 — dense
only for the skeleton; a BM25 index over the SAME chunk IDs is added during
hardening.)
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
    # Chroma metadata values must be str/int/float/bool — drop None.
    return {k: v for k, v in meta.items() if v is not None}


def index_chunks(chunks: list[Chunk], reset: bool = False) -> int:
    col = get_collection(reset=reset)
    if not chunks:
        return 0
    embeddings = embed_texts([c.text for c in chunks])
    col.add(
        ids=[c.metadata.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_clean_meta(c.metadata.model_dump()) for c in chunks],
    )
    return len(chunks)
