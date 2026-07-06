# Support Knowledge Copilot with Verified Citations

A RAG (Retrieval-Augmented Generation) pipeline built from scratch using plain Python —
no frameworks like LangChain or LlamaIndex. Claude Sonnet handles generation, OpenAI
text-embedding-3-small handles embeddings, and ChromaDB serves as the local vector database.

## What This Project Demonstrates

- Building a production-minded RAG pipeline without hiding decisions behind a framework
- Hybrid retrieval combining dense vector search and BM25 sparse retrieval fused with RRF
- Cross-encoder reranking for precise relevance scoring
- Citation verification — every cited chunk is checked against its claim using an LLM judge
- Confidence scoring across four signals — retrieval, citation support, completeness, no-answer
- Evaluation suite measuring retrieval and answer quality separately
- RAG freshness and drift monitoring layer (Project 7, built on top of this)

## Eval Results (hybrid strategy, 20 questions)

| Metric | Value |
|---|---|
| correct source retrieved | 20/20 (100%) |
| mean reciprocal rank | 0.95 |
| answers correct | 17/17 (100%) |
| correct refusals | 3/3 (100%) |
| categories passing | simple_lookup, multi_doc, ambiguous, no_answer, exact_match |

## Current Status
✅ Document ingestion — load, chunk, embed, store in ChromaDB
✅ BM25 sparse index — built alongside Chroma during ingestion
✅ Dense retrieval — vector similarity search with OpenAI embeddings
✅ Sparse retrieval — BM25 keyword search over same chunk IDs
✅ RRF fusion — combines dense and sparse ranked lists
✅ Cross-encoder reranking — top 20 fused → top 5
✅ Grounded generation — Claude Sonnet answers strictly from retrieved context
✅ Citation parsing — extracts (claim, chunk_id) pairs from answer text
✅ Citation verification — LLM judge checks every cited chunk supports its claim
✅ Confidence scoring — retrieval + citation support + completeness + no-answer detection
✅ No-answer handling — detected and confidence hard-capped at 0.40
✅ Evaluation suite — 20 question golden set across 5 categories
✅ Eval CLI — python -m src.eval.run --strategy hybrid
✅ Markdown report with per-question results and failed case analysis
✅ Streamlit dashboard — answer, citations, confidence, sources, strategy toggle
✅ Index manifest — written after every ingestion run
✅ Probe set — 10 probe questions tied to known source sections
⬜ Project 7 — RAG freshness and drift monitor

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Generation | Claude Sonnet (claude-sonnet-4-6) | Strong reasoning, grounded answers |
| Embeddings | OpenAI text-embedding-3-small | High quality, cost effective |
| Vector store | ChromaDB | Zero setup, runs locally, persistent |
| Sparse search | BM25 via rank-bm25 | Catches exact keyword matches dense misses |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Accurate joint question-chunk scoring |
| Judge | Claude Sonnet (claude-sonnet-4-6) | Verifies citations against source chunks |
| Validation | Pydantic | Typed contracts across every module |
| Dashboard | Streamlit | Visual interface with strategy toggle |
| Language | Python 3.11+ | Strong AI and data ecosystem |

## Project Structure

rag-copilot/
├── src/
│   ├── common/        # contracts, embeddings, LLM, judge harness
│   ├── ingest/        # loaders, chunking, vector index, CLI
│   ├── retrieval/     # dense, sparse, fusion, rerank
│   ├── answer/        # prompt, generation, citations, confidence
│   ├── eval/          # golden set, metrics, runner, report
│   └── api/           # FastAPI /ask endpoint (stub)
├── dashboards/
│   └── copilot_app.py # Streamlit dashboard
├── data/
│   ├── corpus/        # source markdown documents
│   ├── chroma/        # ChromaDB vector store (auto-generated, gitignored)
│   └── bm25_index.pkl # BM25 index (auto-generated, gitignored)
├── evals/
│   ├── golden_qa.jsonl  # 20 hand-written Q&A questions across 5 categories
│   ├── probes.jsonl     # drift monitor probe questions (in progress)
│   └── reports/         # eval markdown reports (auto-generated)
├── config/            # settings, chunking, retrieval params
├── ask.py             # end-to-end CLI runner
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

# ask using hybrid retrieval (default) with citation verification
python ask.py "How do I reset my password?"

# ask without citation verification (faster)
python ask.py --no-verify "How do I reset my password?"

# ask using dense only (for comparison)
python ask.py --strategy dense "How do I reset my password?"

# run the eval suite
python -m src.eval.run --no-verify        # fast, no citation verification
python -m src.eval.run                    # full, with citation verification
python -m src.eval.run --strategy dense   # compare dense vs hybrid

# launch the dashboard
streamlit run dashboards/copilot_app.py
```

## Example Output

=== Answer (hybrid) ===
Go to Settings > Security and choose "Reset password." [faq::0]
You will receive a reset link by email which expires after 30 minutes. [faq::0]
=== Citations ===
[faq::0] ✅ supported
claim:    Go to Settings > Security and choose Reset password.
evidence: Go to Settings > Security and choose Reset password.
=== Confidence ===
overall:               0.93
retrieval score:       0.97
citation support rate: 1.00
completeness:          0.75
no answer flag:        False
=== Sources ===
[faq::0]             faq.md             (score 0.0164)
[troubleshooting::0] troubleshooting.md (score 0.0161)
[account_policy::0]  account_policy.md  (score 0.0158)
[onboarding::0]      onboarding.md      (score 0.0154)
[billing::0]         billing.md         (score 0.0157)

## Retrieval Pipeline

question
→ dense retrieval (top 20)         ← semantic meaning
→ sparse BM25 retrieval (top 20)   ← exact keywords
→ RRF fusion                       ← combines both ranked lists
→ cross-encoder rerank             ← top 20 fused → top 5
→ Claude Sonnet generation         ← grounded cited answer
→ citation parsing                 ← extract (claim, chunk_id) pairs
→ citation verification            ← LLM judge checks each citation
→ confidence scoring               ← four signals → one score

## Eval Pipeline

golden_qa.jsonl (20 hand-written questions)
→ run.py feeds each question to the pipeline
→ metrics.py scores each answer:
correct_source_retrieved  (retrieval quality)
reciprocal_rank           (retrieval rank)
answer_correct            (answer quality)
correct_refusal           (no-answer handling)
citation_support_rate     (citation quality)
→ aggregate() combines all scores
→ report.py writes markdown report to evals/reports/

## Architecture Decisions

**No framework** — LangChain and LlamaIndex hide the decisions that matter most
in interviews: chunking strategy, retrieval fusion, citation verification,
confidence scoring. Every decision here is explicit and explainable.

**Two providers** — Claude Sonnet for generation and judging, OpenAI for
embeddings. Anthropic does not offer an embeddings API, so the two concerns
are split deliberately. The embedding layer is isolated in
src/common/embeddings.py so swapping to a local model only touches one file.

**Why hybrid retrieval** — dense retrieval understands meaning but struggles
with exact strings like error codes, email addresses, and product names. BM25
catches those exact matches but misses semantic similarity. RRF combines both
ranked lists using only rank position, not raw scores, which avoids the problem
of cosine similarity and BM25 scores living on incomparable scales.

**Why cross-encoder reranking** — dense retrieval embeds question and chunk
separately. A cross-encoder reads them together as a pair and scores relevance
directly, which is more accurate. It only runs on the top 20 fused candidates
rather than the full corpus to keep latency manageable.

**Citation verification** — an LLM judge checks every cited chunk against its
claim after generation. Supported citations pass through. Unsupported citations
are flagged in the Could Not Verify section. This makes every answer auditable.

**Confidence scoring** — four signals combined with weights: retrieval score
(0.35), citation support rate (0.40), completeness (0.25). If the no-answer
flag fires, overall confidence is hard-capped at 0.40 regardless of other signals.

**Eval design** — retrieval and answer metrics are kept separate deliberately.
A retrieval failure (wrong chunks) and a generation failure (wrong answer)
are different problems that need different fixes. The golden set covers five
categories: simple_lookup, multi_doc, ambiguous, no_answer, and exact_match.

**chunk_hash + embedding_version on every chunk** — both fields are populated
at ingestion time even though nothing reads them yet. They are the substrate
the Project 7 drift monitor needs, reserved now to avoid a retrofit later.

**ChromaDB for local development** — zero setup, persistent, installs with pip.
The vector store is isolated in src/ingest/index.py and src/retrieval/dense.py
so migrating to Qdrant for production only touches those two files.

## Known Issues

- Citation claim text shows the full answer paragraph rather than the individual
  sentence when Claude writes inline tags — minor parsing refinement for hardening.

## Roadmap

- [x] Document ingestion pipeline
- [x] Dense vector retrieval
- [x] BM25 sparse retrieval
- [x] RRF fusion
- [x] Cross-encoder reranking
- [x] Grounded generation with Claude Sonnet
- [x] Citation parsing and verification
- [x] Confidence scoring with breakdown
- [x] No-answer handling
- [x] Golden Q&A eval set — 20 questions across 5 categories
- [x] Eval CLI with markdown report
- [x] Streamlit dashboard
- [x] Index manifest (Project 7 bridge)
- [x] Probe set (Project 7 bridge)
- [ ] Project 7 — RAG freshness and drift monitor

## Note on Project 7

This repo is designed to serve as the foundation for a RAG freshness and drift
monitor (Project 7). The chunk_hash and embedding_version fields on every chunk
are reserved from day one. The index manifest and probe set (Stage B) are the
final two additions before the monitor layer is built on top.