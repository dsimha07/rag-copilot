"""Dense retrieval: embed the query, fetch top-k from Chroma. (FR-8)

Skeleton uses dense only. Sparse (BM25), RRF fusion, and reranking are added
during hardening.
"""
from __future__ import annotations
from src.common import config
from src.common.embeddings import embed_text
from src.common.models import RetrievedChunk, ChunkMetadata
from src.ingest.index import get_collection


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or config.TOP_K
    col = get_collection()
    q_emb = embed_text(question)
    res = col.query(query_embeddings=[q_emb], n_results=top_k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]

    out: list[RetrievedChunk] = []
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        score = None if dist is None else 1.0 - dist  # distance -> rough similarity
        out.append(RetrievedChunk(
            chunk_id=cid,
            text=text,
            metadata=ChunkMetadata(**meta),
            dense_score=score,
        ))
    return out
