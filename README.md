# BioIntel Agent — Agentic RAG for Drug Discovery & Patent Intelligence

> A production-grade, **fully local / open-weights** agentic RAG system that answers
> biomedical drug-discovery and patent questions with **verified, cited** answers over
> literature, clinical trials, drug/target databases, and patents.

BioIntel ingests from **official APIs only** (no HTML scraping), builds a **hybrid
retrieval** index (dense vectors + BM25 with cross-encoder reranking), and runs a
**multi-step LangGraph agent** that plans sub-questions, retrieves, extracts structured
records, detects contradictions across sources, and **verifies every citation** before
returning an answer. Everything runs on a single GPU workstation with no API keys.

---

[Screencast from 2026-07-25 14-32-33.mp4](https://github.com/Arpit-Avasarmol/BioIntel-AgenticRAG/blob/c2a48cb03e9ce0470fedb695ff58c0c019730378/Screencast%20from%202026-07-25%2014-32-33.mp4)

## Table of contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Where the data comes from & how to source it](#3-where-the-data-comes-from--how-to-source-it) ← *data-sourcing guide*
4. [Data licensing](#4-data-licensing)
5. [Quickstart (clone → cited answer)](#5-quickstart-clone--cited-answer)
6. [Model profiles & hardware](#6-model-profiles--hardware)
7. [Configuration](#7-configuration)
8. [Usage](#8-usage)
9. [Observability (opt-in)](#9-observability-opt-in)
10. [Testing & CI](#10-testing--ci)
11. [Project layout](#11-project-layout)
12. [Design decisions & FAQ](#12-design-decisions--faq)

---

## 1. What it does

Ask a question like:

> *"What IL-23 inhibitors are in late-stage trials for Crohn's disease, and what do
> patents claim about anti-IL-23 antibodies?"*

BioIntel will:

1. **Plan** — decompose the question into focused sub-questions.
2. **Retrieve** — hybrid search (dense + sparse) across all indexed sources, fused with
   Reciprocal Rank Fusion, then reranked by a cross-encoder.
3. **Extract** — pull structured records (trials, compounds, targets, patents, findings)
   from the top chunks using schema-constrained LLM decoding.
4. **Synthesize** — write an answer grounded **only** in retrieved sources, citing every
   sentence `[n]`.
5. **Cross-reference** — flag **contradictions** between sources (e.g., a trial reporting
   efficacy vs. a paper reporting a null result).
6. **Verify** — a deterministic, non-LLM check confirms each cited sentence is actually
   supported by its source chunk; unsupported claims trigger one regeneration, and the
   final answer carries a `verified` flag and any warnings.
7. **Audit** — every answer (question, sub-questions, chunks used, citations, model,
   latency, warnings) is written to an append-only audit log.

Key properties: **local & private** (no data leaves the host), **auditable** (raw payloads
archived, citations verifiable), and **swappable** (vector backend and models behind
interfaces).

---

## 2. Architecture

```
                                   ┌──────────────────────────────────────────┐
                                   │  Streamlit chat UI  (:8501)               │
                                   └───────────────────┬──────────────────────┘
                                                       │ HTTP + X-API-Key
                                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI  (:8000)   /query  /query/stream  /ingest  /documents  /health  /metrics       │
│  auth · rate-limit (Redis) · answer cache (Redis) · audit log (Postgres)                 │
└───────────────────────────────────┬─────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌──────────────────────  LangGraph agent  ──────────────────────┐
        │  plan → retrieve → extract → synthesize → contradictions →      │
        │  verify → (regenerate if unverified) → END                      │
        │                          │                                      │
        │                          ▼                                      │
        │             Hybrid retrieval  (RRF + cross-encoder rerank)      │
        │           ┌──────────────┴───────────────┐                      │
        │           ▼                              ▼                       │
        │   Qdrant (dense vectors)      OpenSearch (BM25 sparse)          │
        └────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │  chunk → embed (bge) → upsert
                                     │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Ingestion (official APIs only — NO scraping)                                             │
│  PubMed · PMC OA · ClinicalTrials.gov v2 · ChEMBL · Open Targets · USPTO PatentsView      │
│  (+ optional Google Patents / BigQuery)                                                   │
│  raw payloads → MinIO (S3)      normalized metadata → Postgres                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

Local models (open-weights, no API keys):
  • LLM        → Ollama on host        (qwen2.5:14b-instruct by default)
  • Embeddings → sentence-transformers  (BAAI/bge-large-en-v1.5, 1024-d)
  • Reranker   → FlagEmbedding cross-encoder (BAAI/bge-reranker-v2-m3)

Observability (opt-in profile): Langfuse (LLM traces) · OpenTelemetry → Jaeger · Prometheus · Grafana
```

**Backing services** (docker-compose): PostgreSQL, Redis, MinIO, Qdrant, OpenSearch, and
optionally Milvus (swap-in for Qdrant) and the observability stack.


---

## 3. Where the data comes from & how to source it

**BioIntel does not scrape HTML.** Every document is fetched from an **official,
machine-readable API or bulk-download endpoint**, each of which explicitly permits
programmatic access and carries a clear license. Scraping rendered web pages is fragile,
often against terms of service, and unnecessary here because every source below has a
first-class API.

The corpus is **topic-seeded and config-driven**: you declare a theme and per-source
queries in [`configs/seed_corpus.yaml`](configs/seed_corpus.yaml), and the ingestion
runner pulls a bounded, overlapping slice across all sources so the agent can
cross-reference them. The shipped default theme is *IL-23 / TL1A inhibitors in
inflammatory bowel disease*.

### The four data domains and their official sources

| Domain | Source | Access method (official, no scraping) | Auth |
|---|---|---|---|
| **Literature (abstracts)** | **PubMed** | NCBI **E-utilities** (`esearch` → `efetch`), returns MEDLINE XML | None (optional NCBI API key raises rate limit) |
| **Literature (full text)** | **PMC Open Access subset** | NCBI E-utilities + **PMC OA** service; full text only where the OA license allows | None |
| **Clinical trials** | **ClinicalTrials.gov** | **API v2** (`/api/v2/studies`), JSON | None |
| **Drugs & bioactivities** | **ChEMBL** | **ChEMBL REST API** (targets → molecules → activities), JSON | None |
| **Targets & associations** | **Open Targets Platform** | **GraphQL API** (target–disease associations, known drugs), JSON | None |
| **Patents** | **USPTO PatentsView** | **PatentsView API** (`/patents/query`), JSON | None |
| **Patents (optional)** | **Google Patents Public Data** | **BigQuery** public dataset | GCP service-account creds |

All primary sources are **keyless**. The only source needing credentials is the *optional*
Google Patents BigQuery connector; the no-auth **PatentsView** connector is the default
patent source, so the whole system runs with **zero API keys**.

### How the ingestion works (per source)

Each source has a connector in [`biointel/ingestion/`](biointel/ingestion/) implementing a
common interface:

- **`fetch()`** — paginated, rate-limited, retried (tenacity) calls to the official API.
- **`normalize(raw)`** — maps the raw payload to a canonical `Document` with full
  provenance (IDs, source URL, license, sections). This is the *only* place source-specific
  shape lives.

The runner then archives the **raw payload to MinIO** (auditable) and upserts **normalized
metadata to Postgres** (idempotent by `doc_id`). Indexing chunks each document, embeds the
chunks, and upserts them into Qdrant + OpenSearch.

```bash
# Live ingest of the whole seed corpus from official APIs (no keys needed):
make ingest                       # = biointel ingest --config configs/seed_corpus.yaml && biointel index --all

# Or one source at a time:
biointel ingest --source clinicaltrials --query "Crohn Disease" --max-records 50
biointel index --all
```

**To build a knowledge base on a different topic**, edit `configs/seed_corpus.yaml`
(change `topic` and the per-source queries / disease IDs / target symbols) and re-run
`make ingest`. Nothing else changes.

### Etiquette & compliance built in

- **Rate limiting & backoff** on every connector (respects each API's guidance).
- **Bounded fetches** — a global `max_records_per_source` cap so a demo never runs away.
- **License-aware** — PMC ingests **only** Open Access articles and records the exact
  CC license; non-OA articles are skipped. Every `Document` stores its `license`.
- **No HTML scraping, no paywalled content, no ToS violations.**

### Offline demo (no network at all)

For a laptop demo or CI, [`scripts/seed_demo.py`](scripts/seed_demo.py) loads six bundled
fixtures (one representative record per source), runs them through the **same** normalize →
persist → index path, and gives you a queryable corpus without a single network call:

```bash
make seed        # = python scripts/seed_demo.py
```

---

## 4. Data licensing

Always review each provider's current terms before redistributing data. Summary of the
sources used here:

| Source | Typical license / terms | Notes |
|---|---|---|
| **PubMed** (abstracts/metadata) | U.S. Government work — public domain (metadata) | Abstract text copyright may belong to publishers; store per your use. |
| **PMC Open Access subset** | Creative Commons (CC-BY, CC-BY-NC, CC0, …) per article | BioIntel ingests OA-subset articles only and records the exact license. |
| **ClinicalTrials.gov** | Public information (U.S. NLM) | Free to use; attribution appreciated. |
| **ChEMBL** | **CC BY-SA 3.0** | Attribution + share-alike. |
| **Open Targets** | **CC0 1.0** | Public domain dedication. |
| **USPTO PatentsView** | U.S. Government data — public | Patent documents are public record. |
| **Google Patents Public Data** | Google Cloud public dataset terms | BigQuery billing applies; optional. |

The `license` field on every ingested document preserves this provenance so downstream use
can be filtered by license.

---

## 5. Quickstart (clone → cited answer)

**Prerequisites**

- Docker + Docker Compose
- [Ollama](https://ollama.com) running on the host (for the LLM)
- A GPU with ≥16 GB VRAM for the default `max` profile (or use `lean` on CPU — see
  [§6](#6-model-profiles--hardware))
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (for local dev / CLI)

```bash
# 1. Clone
git clone <your-fork-url> biointel-agent && cd biointel-agent

# 2. Pull the LLM (default profile = max)
ollama pull qwen2.5:14b-instruct

# 3. Configure + install dev tooling
make init                 # creates .env from .env.example, installs deps, pre-commit

# 4. Start the stack (Postgres, Redis, MinIO, Qdrant, OpenSearch, API, UI)
make up

# 5. Apply DB migrations
make migrate

# 6. Seed a demo corpus
make seed                 # OFFLINE fixtures (instant)  — or —
make ingest               # LIVE from official APIs (needs internet, no keys)

# 7. Ask a question
biointel query "What IL-23 inhibitors are used in inflammatory bowel disease?"
#   → cited answer in the terminal, with verified=True and source list

# 8. Or use the UI / API
open http://localhost:8501        # Streamlit chat
open http://localhost:8000/docs   # OpenAPI docs
```

**Verify services any time:**

```bash
make healthcheck          # probes Postgres, Redis, Qdrant, OpenSearch, MinIO, Ollama
```

---

## 6. Model profiles & hardware

All models are **open-weights** and run **locally**. Select a profile with the
`MODEL_PROFILE` env var; any individual env var overrides a profile field. Reference:
[`configs/models.yaml`](configs/models.yaml).

| Profile | LLM (Ollama) | Embeddings | Reranker | VRAM | Notes |
|---|---|---|---|---|---|
| **`max`** (default) | `qwen2.5:14b-instruct` | `bge-large-en-v1.5` (1024-d) | `bge-reranker-v2-m3` | ~16 GB+ | Best answer quality. Plan for 32 GB+ system RAM (OpenSearch + services). |
| **`balanced`** | `llama3.1:8b-instruct-q4_K_M` | `bge-large-en-v1.5` (1024-d) | `bge-reranker-v2-m3` | ~8–12 GB | Strong quality on a mid-range GPU. |
| **`lean`** | `llama3.2:3b` | `bge-small-en-v1.5` (384-d) | `bge-reranker-base` | CPU-OK | Everything runs on CPU; slower LLM. Set `EMBEDDING_DEVICE=cpu` and `RERANKER_DEVICE=cpu`. |

```bash
# Switch profiles (edit .env):
MODEL_PROFILE=balanced
ollama pull llama3.1:8b-instruct-q4_K_M    # pull the matching LLM

# CPU-only box:
MODEL_PROFILE=lean
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
ollama pull llama3.2:3b
```

**GPU vs CPU tradeoffs.** Embeddings and the cross-encoder reranker are the main GPU
consumers besides the LLM; on GPU they use fp16 and are fast, on CPU they still work but
are slower per query. The LLM dominates latency — on CPU, use the `lean` 3B model. Note
that **changing the embedding model changes `EMBEDDING_DIM`**, which requires re-indexing
(`biointel index --reindex`) because the vector dimension must match the Qdrant collection.

---

## 7. Configuration

All settings are environment-driven (Pydantic Settings). Start from
[`.env.example`](.env.example); `make init` copies it to `.env`. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROFILE` | `max` | Selects the model bundle. |
| `API_KEY` | `dev-local-key-change-me` | Required in `X-API-Key` header. **Change it.** |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `60` / `60` | Fixed-window rate limit (Redis). |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `milvus`. |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant. |
| `RETRIEVAL_DENSE_TOP_K` / `_SPARSE_TOP_K` | `25` / `25` | Candidates per modality before fusion. |
| `RERANKER_TOP_K` | `8` | Final chunks after reranking. |
| `CITATION_MIN_OVERLAP` | `0.30` | Min lexical overlap for a citation to count as supported. |
| `AGENT_REGENERATE_ON_UNVERIFIED` | `true` | Regenerate once if verification fails. |
| `OBS_ENABLED` / `LANGFUSE_ENABLED` | `false` | Turn on observability wiring. |

Inspect the resolved config any time:

```bash
biointel info
```

---

## 8. Usage

### CLI

```bash
biointel info                                   # show active config
biointel ingest --config configs/seed_corpus.yaml
biointel ingest --source pubmed --query "IL-23 IBD" --max-records 50
biointel index --all                            # index everything not yet indexed
biointel index --reindex                        # re-chunk + re-embed everything
biointel query "…question…" --top-k 8           # full agent, prints cited answer
biointel init-stores                            # create Qdrant collection + OS index
```

### API

```bash
# Cited answer (JSON)
curl -s http://localhost:8000/query \
  -H "X-API-Key: dev-local-key-change-me" -H "Content-Type: application/json" \
  -d '{"question":"What IL-23 inhibitors are used in IBD?","top_k":8}' | jq

# Streaming (SSE tokens, then a terminal result event)
curl -N http://localhost:8000/query/stream \
  -H "X-API-Key: dev-local-key-change-me" -H "Content-Type: application/json" \
  -d '{"question":"…"}'

# Ingest via API
curl -s http://localhost:8000/ingest \
  -H "X-API-Key: dev-local-key-change-me" -H "Content-Type: application/json" \
  -d '{"source":"clinicaltrials","query":"Crohn Disease","max_records":25,"index":true}'
```

Endpoints: `POST /query`, `POST /query/stream`, `POST /ingest`, `GET /documents`,
`GET /health` (no auth), `GET /metrics` (Prometheus). Full schema at `/docs`.

### Streamlit UI

`http://localhost:8501` — chat with source filters; renders citations, contradictions, a
verification badge, and a trace expander. Configure via `BIOINTEL_API_URL` /
`BIOINTEL_API_KEY`.

---

## 9. Observability (opt-in)

Observability lives behind a compose profile so the base stack stays lean:

```bash
make up-obs      # = docker compose --profile observability up -d
# also set OBS_ENABLED=true and LANGFUSE_ENABLED=true in .env
```

| Tool | URL | What you get |
|---|---|---|
| **Langfuse** | http://localhost:3000 | LLM call traces, prompts, token usage, per-step spans. |
| **Jaeger** | http://localhost:16686 | Distributed traces (OpenTelemetry) across API + agent. |
| **Prometheus** | http://localhost:9090 | Metrics: query/ingest counters, query & retrieval latency. |
| **Grafana** | http://localhost:3001 | Dashboards over Prometheus. |

The API always exposes `/metrics`; the tracing exporters activate only when enabled.

---

## 10. Testing & CI

```bash
make test        # offline unit/contract tests (no services, no network, no ML weights)
make test-all    # includes integration tests (requires services up)
make lint        # ruff check
make fmt         # ruff format
```

The offline suite covers connector normalization (against fixtures), chunking determinism,
RRF math + reranker degradation + hybrid orchestration, citation verification, the full
agent pipeline (happy path, regeneration, honest "no results" refusal), the API contract
(auth, cache, SSE, validation), and Alembic migrations up/down on SQLite.

**GitHub Actions CI** (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format
--check`, and the offline `pytest` on every push. **pre-commit** (`.pre-commit-config.yaml`)
runs ruff + hygiene hooks locally.

> The offline tests intentionally avoid heavy ML dependencies (torch,
> sentence-transformers, FlagEmbedding, langgraph) so CI is fast and hermetic; those are
> exercised at runtime on the workstation.

---

## 11. Project layout

```
biointel/
  common/       config (Pydantic Settings), logging, schemas, text utils
  ingestion/    7 source connectors + runner (fetch → normalize → persist)
  indexing/     chunker, embedder (bge), indexing runner
  retrieval/    base interface, Qdrant + OpenSearch + Milvus stores, RRF fusion, reranker, factory
  agent/        LLM client (Ollama), prompts, structured schemas, LangGraph graph + nodes,
                contradiction detection, deterministic citation verification
  api/          FastAPI app, request/response schemas, auth + rate-limit deps
  ui/           Streamlit chat client
  db/           SQLAlchemy models, repository, session, Redis cache, MinIO storage
  obs/          telemetry (Prometheus + OTel) and Langfuse tracing
  cli.py        Typer CLI (info / ingest / index / query / init-stores)
configs/        seed_corpus.yaml (topic-seeded corpus), models.yaml, prometheus.yml
scripts/        seed_demo.py (offline seed), healthcheck.py, package_zip.py
alembic/        migrations
tests/          offline unit/contract tests + fixtures
docker-compose.yml · Dockerfile · Makefile · pyproject.toml
```

---

## 12. Design decisions & FAQ

**Why official APIs instead of scraping?** Every domain here has a first-class,
machine-readable API with a clear license. APIs are stable, respectful of the provider, and
carry provenance/licensing metadata that scraping loses. Scraping rendered HTML is brittle
and frequently violates terms of service.

**Why hybrid retrieval + reranking?** Dense vectors capture semantics; BM25 captures exact
terms (gene symbols, NCT IDs, drug names) that embeddings can miss. RRF fuses both without
score-scale headaches, and a cross-encoder reranker sharpens the final ordering.

**Why verify citations deterministically?** LLMs can cite plausibly but wrongly. A
non-LLM lexical check (marker presence + content-word overlap against the exact source
chunk) makes "verified" mean something auditable, and drives an automatic regeneration when
a claim isn't grounded.

**Why Qdrant + OpenSearch (with Milvus as a swap-in)?** Qdrant is a fast, simple dense
vector DB; OpenSearch provides mature BM25. Both sit behind a `VectorStore` /
keyword-store interface, so Milvus can replace Qdrant via `VECTOR_BACKEND=milvus` with no
code changes.

**Do I need any API keys or cloud accounts?** No. All primary sources are keyless and all
models are local. Only the *optional* Google Patents BigQuery connector needs GCP creds.

**Can I run it without a GPU?** Yes — use `MODEL_PROFILE=lean` with
`EMBEDDING_DEVICE=cpu` / `RERANKER_DEVICE=cpu`. The LLM will be slower but the pipeline is
identical.

---

*Built as a portfolio project demonstrating production-grade agentic RAG: multi-step LLM
orchestration, hybrid retrieval with reranking, structured extraction, contradiction
detection, deterministic citation verification, and full observability — all local and
open-weights.*
