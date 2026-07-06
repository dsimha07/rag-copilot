"""Index manifest — read/write the record of what was indexed.

Written every time ingestion runs. Records embedding model, chunking
strategy, per-doc content hashes, chunk counts, and timestamp.

Project 7's drift monitor reads this to know what the index looked
like before a document changed — it's the baseline for comparison.
"""
from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from src.common import config

MANIFEST_PATH = config.ROOT / "data" / "index_manifest.json"


class DocManifestEntry(BaseModel):
    doc_id: str
    source_path: str
    content_hash: str
    last_modified: str
    n_chunks: int


class IndexManifest(BaseModel):
    manifest_version: str = "1.0"
    created_at: str
    embedding_model: str
    embedding_version: str
    chunking_strategy: str
    chunking_params: dict
    doc_count: int
    chunk_count: int
    vector_collection: str
    docs: list[DocManifestEntry]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    docs: list[dict],
    chunks_per_doc: dict[str, int],
) -> IndexManifest:
    doc_entries = []
    for doc in docs:
        path = Path(doc["source_path"])
        doc_entries.append(DocManifestEntry(
            doc_id=doc["doc_id"],
            source_path=doc["source_path"],
            content_hash=_hash_text(doc["text"]),
            last_modified=datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat() if path.exists() else "",
            n_chunks=chunks_per_doc.get(doc["doc_id"], 0),
        ))

    return IndexManifest(
        created_at=datetime.now().isoformat(),
        embedding_model=config.EMBEDDING_MODEL,
        embedding_version=config.EMBEDDING_MODEL,
        chunking_strategy="fixed_overlap",
        chunking_params={
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        doc_count=len(docs),
        chunk_count=sum(chunks_per_doc.values()),
        vector_collection=config.COLLECTION,
        docs=doc_entries,
    )


def write_manifest(manifest: IndexManifest) -> Path:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8"
    )
    return MANIFEST_PATH


def read_manifest() -> IndexManifest | None:
    if not MANIFEST_PATH.exists():
        return None
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return IndexManifest(**data)