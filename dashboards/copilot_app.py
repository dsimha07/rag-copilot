"""Streamlit dashboard for the Support Knowledge Copilot. (FR-21)

Run with:
    streamlit run dashboards/copilot_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.retrieval.dense import retrieve as dense_retrieve
from src.retrieval.sparse import sparse_retrieve
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank
from src.answer.generate import answer_question
from src.common.models import RetrievedChunk, ChunkMetadata
from src.ingest.index import get_collection

st.set_page_config(
    page_title="Support Knowledge Copilot",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_chroma_collection():
    return get_collection()


def hybrid_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    dense_results = dense_retrieve(question, top_k=20)
    sparse_results = sparse_retrieve(question, top_k=20)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    col = get_chroma_collection()
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


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📚 Support Knowledge Copilot")
st.caption("Answers grounded in internal documentation with verified citations.")

with st.sidebar:
    st.header("Settings")
    strategy = st.radio(
        "Retrieval strategy",
        ["hybrid", "dense"],
        index=0,
        help="Hybrid combines dense vector search + BM25 keyword search."
    )
    verify = st.toggle(
        "Citation verification",
        value=True,
        help="Check every cited chunk supports its claim using an LLM judge."
    )
    st.divider()
    st.caption("hybrid = dense + BM25 + RRF fusion + cross-encoder rerank")

question = st.text_input(
    "Ask a question",
    placeholder="How do I reset my password?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        if strategy == "dense":
            chunks = dense_retrieve(question)
        else:
            chunks = hybrid_retrieve(question)

        result = answer_question(
            question=question,
            chunks=chunks,
            strategy=strategy,
            verify=verify,
        )

    # answer
    st.subheader("Answer")
    st.markdown(result.answer)

    # citations and confidence side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Citations")
        if result.citations:
            for c in result.citations:
                status = "✅ supported" if c.supported else "❌ unsupported"
                with st.expander(f"[{c.chunk_id}] {status}"):
                    st.markdown(f"**Claim:** {c.claim}")
                    if c.evidence_span:
                        st.markdown(f"**Evidence:** {c.evidence_span}")
        else:
            st.caption("No citations parsed." if verify else "Verification skipped.")

        if result.unverified:
            st.warning("**Could not verify:**\n" + result.unverified)

    with col2:
        st.subheader("Confidence")
        if result.confidence:
            c = result.confidence
            st.metric("Overall", f"{c.overall:.2f}")
            st.progress(c.overall)
            st.markdown(f"- Retrieval score: **{c.retrieval_score:.2f}**")
            st.markdown(f"- Citation support: **{c.citation_support_rate:.2f}**")
            st.markdown(f"- Completeness: **{c.completeness:.2f}**")
            if c.no_answer_flag:
                st.error("⚠️ No answer detected — confidence capped at 0.40")

    # sources
    st.subheader("Sources")
    for chunk in result.retrieved_chunks:
        score = chunk.fused_score if chunk.fused_score is not None else chunk.dense_score
        score_str = f"{score:.4f}" if score is not None else "n/a"
        with st.expander(f"[{chunk.chunk_id}] {chunk.metadata.source_name} (score {score_str})"):
            st.text(chunk.text[:500])