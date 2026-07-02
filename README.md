# Support Knowledge Copilot with Verified Citations

A RAG (Retrieval-Augmented Generation) pipeline built from scratch using plain Python — 
no frameworks like LangChain or LlamaIndex. Claude Sonnet handles generation, OpenAI 
text-embedding-3-small handles embeddings, and ChromaDB serves as the local vector database.

## What This Project Demonstrates

- Building a production-minded RAG pipeline without hiding decisions behind a framework
- Hybrid retrieval combining dense vector search and BM25 sparse retrieval fused with RRF
- Cross-encoder reranking for precise relevance scoring
- Citation verification — checking every cited chunk actually supports its claim (in progress)
- Evaluation suite measuring retrieval and answer quality separately (in progress)
- RAG freshness and drift monitoring layer (Project 7, built on top of this)

## Current Status

✅ Document ingestion — load, chunk, embed, store in ChromaDB

✅ BM25 sparse index — built alongside Chroma during ingestion

✅ Dense retrieval — vector similarity search with OpenAI embeddings

✅ Sparse retrieval — BM25 keyword search over same chunk IDs

✅ RRF fusion — combines dense and sparse ranked lists

✅ Cross-encoder reranking — top 20 fused → top 5

✅ Grounded generation — Claude Sonnet answers strictly from retrieved context

✅ Hybrid pipeline verified — cites two sources vs dense-only citing one

⬜ Citation parsing and verification

⬜ Confidence scoring and no-answer handling

⬜ Evaluation suite

⬜ Streamlit dashboard

⬜ FastAPI /ask endpoint


## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Generation | Claude Sonnet (claude-sonnet-4-6) | Strong reasoning, grounded answers |
| Embeddings | OpenAI text-embedding-3-small | High quality, cost effective |
| Vector store | ChromaDB | Zero setup, runs locally, persistent |
| Sparse search | BM25 via rank-bm25 | Catches exact keyword matches dense misses |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Accurate joint question-chunk scoring |
| Validation | Pydantic | Typed contracts across every module |
| Language | Python 3.11+ | Strong AI and data ecosystem |

## Project Structure

rag-copilot/

├── src/

│   ├── common/        # shared contracts, embeddings, LLM, judge

│   ├── ingest/        # loaders, chunking, vector index, CLI

│   ├── retrieval/     # dense, sparse, fusion, rerank

│   ├── answer/        # prompt, generation, citations (stub), confidence (stub)

│   ├── eval/          # golden set, metrics, runner, report (stubs)

│   └── api/           # FastAPI /ask endpoint (stub)

├── dashboards/        # Streamlit copilot view (stub)

├── data/

│   ├── corpus/        # source markdown documents

│   ├── chroma/        # ChromaDB vector store (auto-generated, gitignored)

│   └── bm25_index.pkl # BM25 index (auto-generated, gitignored)

├── evals/

│   ├── golden_qa.jsonl  # hand-written Q&A eval set (in progress)

│   └── probes.jsonl     # drift monitor probe questions (in progress)

├── config/            # settings, chunking, retrieval params

├── ask.py             # end-to-end runner

└── tests/             # unit tests


## Setup

```bash
# clone the repo
git clone https://github.com/yourusername/rag-copilot.git
cd rag-copilot

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# configure environment
cp .env.example .env
# add your keys to .env:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

## Usage

```bash
# ingest the corpus (run once, or after adding new documents)
python -m src.ingest.cli --rebuild

# ask using hybrid retrieval (default)
python ask.py "How do I reset my password?"

# ask using dense only (for comparison)
python ask.py --strategy dense "How do I reset my password?"
```

## Example Output

=== Answer (hybrid) ===

To reset your password, go to Settings > Security and choose

"Reset password". A reset link will be sent to your email and

expires after 30 minutes [faq::0]. If you don't receive it,

check your spam folder — emails come from no-reply@acme.example [faq::0].

Repeated failed logins lock the account for 15 minutes [troubleshooting::0].
=== Sources ===

[faq::0]             faq.md             (score 0.0164)

[troubleshooting::0] troubleshooting.md (score 0.0161)

[account_policy::0]  account_policy.md  (score 0.0158)

[billing::0]         billing.md         (score 0.0157)

[onboarding::0]      onboarding.md      (score 0.0154)

## Retrieval Pipeline

question

→ dense retrieval (top 20)     ← semantic meaning

→ sparse BM25 retrieval (top 20) ← exact keywords

→ RRF fusion                   ← combines both ranked lists

→ cross-encoder rerank         ← top 20 fused → top 5

→ Claude Sonnet generation     ← grounded cited answer

## Architecture Decisions

**No framework** — LangChain and LlamaIndex hide the decisions that matter most
in interviews: chunking strategy, retrieval fusion, citation verification,
confidence scoring. Every decision here is explicit and explainable.

**Two providers** — Claude Sonnet for generation, OpenAI for embeddings.
Anthropic does not offer an embeddings API, so the two concerns are split
deliberately. The embedding layer is isolated in src/common/embeddings.py
so swapping to a local model (bge-small) only touches one file.

**Why hybrid retrieval** — dense retrieval understands meaning but struggles
with exact strings like error codes, email addresses, and product names. BM25
catches those exact matches but misses semantic similarity. RRF combines both
ranked lists using only rank position, not raw scores, which avoids the problem
of cosine similarity and BM25 scores living on incomparable scales.

**Why cross-encoder reranking** — dense retrieval embeds question and chunk
separately. A cross-encoder reads them together as a pair and scores relevance
directly, which is more accurate. It's slower, so it only runs on the top 20
fused candidates rather than the full corpus.

**chunk_hash + embedding_version on every chunk** — both fields are populated
at ingestion time even though nothing reads them yet. They are the substrate
the Project 7 drift monitor needs, reserved now to avoid a retrofit later.

**ChromaDB for local development** — zero setup, persistent, installs with pip.
The vector store is isolated in src/ingest/index.py and src/retrieval/dense.py
so migrating to Qdrant for production only touches those two files.

## Roadmap

- [x] Document ingestion pipeline
- [x] Dense vector retrieval
- [x] BM25 sparse retrieval
- [x] RRF fusion
- [x] Cross-encoder reranking
- [x] Grounded generation with Claude Sonnet
- [ ] Citation parsing and verification
- [ ] Confidence scoring with breakdown
- [ ] No-answer handling
- [ ] Golden Q&A eval set and metrics
- [ ] Streamlit dashboard
- [ ] FastAPI /ask endpoint
- [ ] Project 7 — RAG freshness and drift monitor

## Note on Project 7

This repo is designed to serve as the foundation for a RAG freshness and drift
monitor (Project 7). The chunk_hash and embedding_version fields on every chunk,
and the index manifest written at ingestion time, are the hooks that monitor
needs. No retrofit required when that layer is added.