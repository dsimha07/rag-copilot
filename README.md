# Support Knowledge Copilot with Verified Citations

A RAG (Retrieval-Augmented Generation) pipeline built from scratch using plain Python — 
no frameworks like LangChain or LlamaIndex. Claude Sonnet handles generation, OpenAI 
text-embedding-3-small handles embeddings, and ChromaDB serves as the local vector database.

## What This Project Demonstrates

- Building a production-minded RAG pipeline without hiding decisions behind a framework
- Hybrid retrieval combining dense vector search and sparse BM25 (in progress)
- Citation verification — checking every cited chunk actually supports its claim (in progress)
- Evaluation suite measuring retrieval and answer quality separately (in progress)
- RAG freshness and drift monitoring layer (Project 7, built on top of this)

## Current Status

Walking skeleton verified end to end:
- ✅ Document ingestion — load, chunk, embed, store in ChromaDB
- ✅ Dense retrieval — vector similarity search with OpenAI embeddings
- ✅ Grounded generation — Claude Sonnet answers strictly from retrieved context with citations
- ⬜ BM25 sparse retrieval + RRF fusion (hybrid retrieval)
- ⬜ Cross-encoder reranking
- ⬜ Citation verification
- ⬜ Confidence scoring and no-answer handling
- ⬜ Evaluation suite
- ⬜ Streamlit dashboard

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Generation | Claude Sonnet (claude-sonnet-4-6) | Strong reasoning, grounded answers |
| Embeddings | OpenAI text-embedding-3-small | High quality, cost effective |
| Vector store | ChromaDB | Zero setup, runs locally, persistent |
| Validation | Pydantic | Typed contracts across every module |
| Language | Python 3.11+ | Strong AI and data ecosystem |

## Project Structure

rag-copilot/

├── src/

│   ├── common/        # shared contracts, embeddings, LLM, judge

│   ├── ingest/        # loaders, chunking, vector index, CLI

│   ├── retrieval/     # dense, sparse (stub), fusion (stub), rerank (stub)

│   ├── answer/        # prompt, generation, citations (stub), confidence (stub)

│   ├── eval/          # golden set, metrics, runner, report (all stubs)

│   └── api/           # FastAPI /ask endpoint (stub)

├── dashboards/        # Streamlit copilot view (stub)

├── data/

│   ├── corpus/        # source markdown documents

│   └── chroma/        # ChromaDB vector store (auto-generated, gitignored)

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

# ask a question
python ask.py "How do I reset my password?"
```

## Example Output

=== Answer ===

To reset your password, go to Settings > Security and choose

"Reset password". A reset link will be sent to your email and

expires after 30 minutes [faq::0]. If you don't receive it,

check your spam folder — emails come from no-reply@acme.example [faq::0].
=== Sources ===

[faq::0]             faq.md             (score 0.049)

[troubleshooting::0] troubleshooting.md (score -0.176)

[account_policy::0]  account_policy.md  (score -0.479)

[billing::0]         billing.md         (score -0.647)


## Architecture Decisions

**No framework** — LangChain and LlamaIndex hide the decisions that matter most 
in interviews: chunking strategy, retrieval fusion, citation verification, 
confidence scoring. Every decision here is explicit and explainable.

**Two providers** — Claude Sonnet for generation, OpenAI for embeddings. 
Anthropic does not offer an embeddings API, so the two concerns are split 
deliberately. The embedding layer is isolated in src/common/embeddings.py 
so swapping to a local model (bge-small) only touches one file.

**chunk_hash + embedding_version on every chunk** — both fields are populated 
at ingestion time even though nothing reads them yet. They are the substrate 
the Project 7 drift monitor needs, reserved now to avoid a retrofit later.

**ChromaDB for local development** — zero setup, persistent, installs with pip. 
The vector store is isolated in src/ingest/index.py and src/retrieval/dense.py 
so migrating to Qdrant for production only touches those two files.

## Roadmap

- [ ] BM25 sparse retrieval alongside dense
- [ ] RRF fusion combining both result lists
- [ ] Cross-encoder reranking top-20 → top-5
- [ ] Citation verification (check cited chunk supports its claim)
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