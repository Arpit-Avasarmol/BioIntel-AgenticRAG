# BioIntel Agent — Interview Preparation Guide

> A structured reference for explaining the project, its scripts, workflows, execution steps, service health checks, and how new topics enter the corpus. Use this alongside [README.md](README.md).

---

## Table of contents

1. [Project in one sentence](#1-project-in-one-sentence)
2. [High-level architecture](#2-high-level-architecture)
3. [Every script and when it runs](#3-every-script-and-when-it-runs)
4. [Config files](#4-config-files-not-scripts-but-interview-relevant)
5. [Core Python modules](#5-core-python-modules-what-each-package-does)
6. [End-to-end workflows](#6-end-to-end-workflows)
7. [How to execute the full repository](#7-how-to-execute-the-full-repository)
8. [How to check each service](#8-how-to-check-each-service)
9. [New corpus for a fresh query](#9-new-corpus-for-a-fresh-query--how-it-actually-works)
10. [Likely interview questions & answers](#10-likely-interview-questions--strong-answers)
11. [Quick reference cheat sheet](#11-quick-reference-cheat-sheet)

---

## 1. Project in one sentence

BioIntel Agent is a **local, agentic RAG system** for drug discovery and patent intelligence. It ingests biomedical data from **official APIs only** (no scraping), indexes it into **hybrid search** (dense vectors + BM25), and runs a **multi-step LangGraph agent** that plans, retrieves, extracts structured records, synthesizes cited answers, detects contradictions, and **verifies citations** before returning results.

---

## 2. High-level architecture

```
                                   ┌──────────────────────────────────────────┐
                                   │  Streamlit chat UI  (:8501)               │
                                   └───────────────────┬──────────────────────┘
                                                       │ HTTP + X-API-Key
                                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI  (:8000)   /query  /query/stream  /ingest  /documents  /health  /metrics   │
│  auth · rate-limit (Redis) · answer cache (Redis) · audit log (Postgres)                │
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
```

**Key design point:** There is **one shared corpus**, not a new database per query. Fresh topics are added by **ingesting + indexing** new documents into the same Postgres / Qdrant / OpenSearch stack.

### Data stores (what lives where)

| Store | Holds |
|-------|-------|
| **PostgreSQL** | Document metadata, chunk metadata, audit log, chat sessions |
| **MinIO** | Raw/normalized JSON archives (auditable) |
| **Qdrant** | Embedding vectors (dense search) |
| **OpenSearch** | Searchable text (BM25 sparse search) |
| **Redis** | Answer cache + API rate limiting |
| **Ollama (host)** | Local LLM inference |

---

## 3. Every script and when it runs

### 3.1 `scripts/` (standalone Python scripts)

| Script | What it does | When you run it |
|--------|--------------|-----------------|
| **`scripts/seed_demo.py`** | Loads 6 offline JSON/XML fixtures (PubMed, PMC, trials, ChEMBL, Open Targets, PatentsView). Normalizes via real connectors, persists to Postgres/MinIO, chunks, embeds, upserts to Qdrant + OpenSearch. | `make seed` — first demo, CI, or when you want **no network**. Requires `make up` + `make migrate`. Does **not** need Ollama. |
| **`scripts/healthcheck.py`** | Probes Postgres, Redis, Qdrant, OpenSearch, MinIO (required), and Ollama (optional). Exits non-zero if required services are down. | `make healthcheck` — after `make up`, before queries, or in smoke tests. Use `--require-ollama` to also fail if LLM is down. |
| **`scripts/package_zip.py`** | Creates a clean distributable `.zip` of the repo (excludes `.venv`, caches, `.git`). | `make zip` — packaging/distribution, not part of normal runtime. |

#### `seed_demo.py` step-by-step

1. Reads fixtures from `tests/fixtures/` (one record per source).
2. Runs each through its connector's `normalize()` (same code path as live ingest).
3. Persists documents via `_persist()` → Postgres + MinIO.
4. Calls `run_indexing(all_docs=True)` → chunk, embed, upsert Qdrant + OpenSearch.

#### `healthcheck.py` step-by-step

1. Reads endpoints from `settings` (`.env`).
2. Probes each service with a lightweight request (`SELECT 1`, `ping`, `/readyz`, etc.).
3. Prints OK/DOWN table; exits `1` if any **required** service fails.
4. Ollama is optional by default (only needed for queries, not ingest/index).

### 3.2 Makefile commands (orchestration layer)

| Command | Executes | When |
|---------|----------|------|
| **`make init`** | Copies `.env.example` → `.env`, creates `.venv`, installs deps with `uv`, optional pre-commit | **Once** after clone |
| **`make up`** | `docker compose up -d` — Postgres, Redis, MinIO, Qdrant, OpenSearch, API, UI | Every time you start the stack |
| **`make up-obs`** | Same + Jaeger, Prometheus, Grafana, Langfuse | When you need observability |
| **`make down`** | Stops containers (keeps volumes) | Shutdown |
| **`make clean`** | Stops containers **and deletes all data volumes** | Full reset / wipe corpus |
| **`make migrate`** | `alembic upgrade head` — creates/updates Postgres tables | After first `make up` or schema changes |
| **`make seed`** | `scripts/seed_demo.py` | Offline demo corpus |
| **`make ingest`** | `biointel ingest --config configs/seed_corpus.yaml` then `biointel index --all` | Live ingest of IL-23/IBD theme from APIs |
| **`make query q="..."`** | Full agent query from CLI | Testing / demos |
| **`make healthcheck`** | `scripts/healthcheck.py` | Verify services |
| **`make test`** | Offline pytest (no services) | CI / dev |
| **`make test-all`** | Full pytest including integration | When stack is up |
| **`make api` / `make ui`** | Run API or Streamlit on host (not Docker) | Local dev without rebuilding containers |
| **`make lint` / `make fmt`** | Ruff check / format | Code quality |
| **`make zip`** | `scripts/package_zip.py` | Distribution |

**Note:** `make query` and host-side CLI commands set `HOST_ENV` so services resolve to `localhost` (containers use docker DNS names from `.env`).

### 3.3 CLI (`biointel` — entry point in `biointel/cli.py`)

| Command | What it does | When |
|---------|--------------|------|
| **`biointel info`** | Prints active config (models, URLs, DB) | Debug config |
| **`biointel ingest`** | Fetches from one source or YAML config → Postgres + MinIO | Manual / batch corpus build |
| **`biointel index`** | Chunk → embed → Qdrant + OpenSearch | After ingest, or `--reindex` to rebuild vectors |
| **`biointel query`** | Full agent pipeline + printed answer | CLI queries |
| **`biointel init-stores`** | Creates Qdrant collection + OpenSearch index if missing | First-time store setup |

**CLI flags worth knowing:**

```bash
biointel ingest --config configs/seed_corpus.yaml
biointel ingest --source pubmed --query "IL-23 IBD" --max-records 50
biointel index --all
biointel index --reindex          # wipe vector + keyword stores, re-embed everything
biointel query "your question" --top-k 8 --auto-ingest / --no-auto-ingest
```

---

## 4. Config files (not scripts, but interview-relevant)

| File | Role |
|------|------|
| **`configs/seed_corpus.yaml`** | Defines topic + per-source queries for **static** seeding (IL-23/IBD default). Used by `make ingest`. |
| **`configs/models.yaml`** | Documents model profiles (`max`, `balanced`, `lean`). |
| **`configs/prometheus.yml`** | Prometheus scrape config for observability profile. |
| **`.env`** | All runtime settings (DB URLs, API key, auto-ingest flags, devices). |
| **`alembic.ini` + `alembic/versions/`** | Database migrations for Postgres schema. |
| **`docker-compose.yml`** | Defines all backing services + API/UI containers. |
| **`Dockerfile`** | Builds the API/UI image (Python 3.12 + app code). |
| **`pyproject.toml`** | Package metadata, dependencies, pytest/ruff config. |
| **`.pre-commit-config.yaml`** | Local hooks (ruff) before commit. |

---

## 5. Core Python modules (what each package does)

### `biointel/ingestion/` — Fetch & normalize

| Module | Source | API |
|--------|--------|-----|
| `pubmed.py` | Literature abstracts | NCBI E-utilities |
| `pmc.py` | Full-text OA papers | PMC OA |
| `clinicaltrials.py` | Trials | ClinicalTrials.gov v2 |
| `chembl.py` | Compounds / bioactivities | ChEMBL REST |
| `opentargets.py` | Target–disease links | Open Targets GraphQL |
| `patentsview.py` | US patents | USPTO PatentsView / PatentSearch API |
| `google_patents.py` | Patents (optional) | BigQuery |
| `base.py` | `BaseConnector`, `HttpClient` (rate limit, retry) | Shared HTTP layer |
| `runner.py` | Registry + `ingest_single` / `ingest_from_config` | Orchestrates connectors |
| **`dynamic.py`** | **Query-driven auto-ingest** | Runs before agent when `AGENT_AUTO_INGEST=true` |

**Connector pattern:** `fetch()` → raw records → `normalize()` → canonical `Document` → `_persist()`.

**`dynamic.py` highlights:**

- `heuristic_ingest_plan()` — extracts drug/gene terms, builds PubMed/PatentsView queries.
- `ensure_corpus_for_question()` — ingest + index before retrieval.
- `infer_retrieval_context()` — sets `doc_type` filters and `required_terms` for retrieval.
- Modes: `always` (every query) or `if_needed` (only when corpus lacks topic).

### `biointel/indexing/` — Chunk & embed

| Module | Role |
|--------|------|
| `chunker.py` | Splits documents into deterministic chunks |
| `embedder.py` | BGE embeddings (sentence-transformers), CPU/GPU with fallback |
| `runner.py` | `run_indexing()` — reads Postgres, embeds, upserts Qdrant + OpenSearch |

### `biointel/retrieval/` — Search

| Module | Role |
|--------|------|
| `qdrant_store.py` | Dense vector search |
| `opensearch_store.py` | BM25 sparse search |
| `milvus_store.py` | Optional Qdrant replacement (`VECTOR_BACKEND=milvus`) |
| `hybrid.py` | Parallel dense + sparse → **RRF fusion** → cross-encoder rerank |
| `reranker.py` | BGE cross-encoder reranking |
| `factory.py` | `get_vector_store()`, `get_keyword_store()` |
| `base.py` | Store interfaces + chunk payload helpers |

**Hybrid retrieval pipeline:**

1. Dense search (Qdrant) and sparse search (OpenSearch) run **in parallel**.
2. **Reciprocal Rank Fusion (RRF)** merges ranked lists (rank-based, no score normalization).
3. Cross-encoder reranker scores fused candidates.
4. Top-k chunks returned to the agent.

### `biointel/agent/` — Reasoning pipeline

| Module | Role |
|--------|------|
| `graph.py` | LangGraph wiring + `run_agent()` entry point |
| `nodes.py` | `plan` → `retrieve` → `extract` → `synthesize` → `contradictions` → `verify` → optional `regenerate` |
| `llm.py` | Ollama client + JSON schema decoding |
| `prompts.py` | System prompts for planner, synthesizer, ingest planner, etc. |
| `structures.py` | Pydantic schemas (TrialRecord, PatentRecord, QueryPlan, DynamicIngestPlan, …) |
| `verify.py` | **Deterministic** citation verification (lexical overlap, not LLM) |
| `state.py` | TypedDict state passed between nodes |

**Agent graph:**

```
plan → retrieve → extract → synthesize → contradictions → verify
                              ↑                              |
                              |         (unverified)         |
                              +-------- regenerate <--------+
                                                           |
                                              (verified) → END
```

### `biointel/api/` — HTTP layer

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness + Ollama status |
| `GET /metrics` | Prometheus |
| `POST /query` | Full agent answer (JSON) |
| `POST /query/stream` | SSE streaming |
| `POST /ingest` | Trigger ingest + optional index |
| `GET /documents` | List ingested docs |

Auth: `X-API-Key`. Rate limit + answer cache via Redis. Audit log in Postgres.

### `biointel/ui/` — Streamlit chat

- Chat UI at `:8501`.
- Calls API with `BIOINTEL_API_URL` and `BIOINTEL_API_KEY`.
- Shows citations, contradictions, verification badge, auto-fetch toggle.

### `biointel/db/` — Persistence layer

| Module | Role |
|--------|------|
| `models.py` | SQLAlchemy ORM (`DocumentRecord`, `ChunkRecord`, `AuditLog`, …) |
| `repository.py` | Upsert documents, mark indexed, audit writes |
| `session.py` | DB session management |
| `storage.py` | MinIO raw archive |
| `cache.py` | Redis answer cache |

### `biointel/obs/` — Observability (opt-in)

| Module | Role |
|--------|------|
| `telemetry.py` | Prometheus metrics, OTel wiring |
| `tracing.py` | Langfuse integration |

### `biointel/common/` — Shared utilities

| Module | Role |
|--------|------|
| `config.py` | Pydantic Settings from `.env` |
| `schemas.py` | `Document`, `Chunk`, `Citation`, `AgentAnswer`, enums |
| `logging.py` | Structured logging |
| `text.py` | Text cleaning utilities |
| `devices.py` | Torch device resolution (CPU fallback) |

---

## 6. End-to-end workflows

### Workflow A — First-time setup (recommended path)

```text
1. git clone → cd biointel-agent
2. ollama pull qwen2.5:14b-instruct
3. make init                    # .env + .venv
4. make up                      # Docker stack
5. make migrate                 # Postgres tables
6. make healthcheck             # all services OK
7. make seed                    # offline 6-doc demo OR make ingest (live)
8. make query q="your question" # or UI at :8501
```

### Workflow B — Live static corpus (`make ingest`)

```text
configs/seed_corpus.yaml
    → ingest_from_config() for each enabled source
    → fetch from official APIs (rate-limited, retried)
    → normalize → Document
    → MinIO (raw JSON) + Postgres (metadata)
    → run_indexing()
    → chunk → embed → Qdrant + OpenSearch
```

### Workflow C — Per-query auto-ingest (fresh topics e.g. Azithromycin)

Triggered when `AGENT_AUTO_INGEST=true` and `AGENT_AUTO_INGEST_MODE=always` (current default).

```text
User asks: "What patents claim Azithromycin?"
    → run_agent() calls ensure_corpus_for_question()
    → heuristic/LLM ingest plan:
        - patent_query = "Azithromycin"
        - pubmed_query = "Azithromycin"[Title/Abstract] AND patent[Title/Abstract]
    → ingest_single() per source (PatentsView first for patent questions)
    → run_indexing(all_docs=False)  # only new docs
    → infer_retrieval_context() sets filters + required_terms
    → agent: plan → retrieve (hybrid + term filter) → extract → synthesize → verify
```

**Important:** Auto-ingest **extends the existing corpus**, it does not create a separate database.

### Workflow D — Agent query (after corpus exists)

```text
plan_node           LLM splits question into sub-questions
retrieve_node       Hybrid search per sub-question (Qdrant + OpenSearch → RRF → rerank)
                    Optional: filter chunks by required_terms (e.g. "azithromycin")
extract_node        Schema-constrained JSON per chunk (trial/compound/patent/paper)
synthesize_node     Grounded answer with [n] citations
contradiction_node  Cross-source conflict detection
verify_node         Lexical check: each citation must support its sentence
regenerate_node     One retry if verify fails (when AGENT_REGENERATE_ON_UNVERIFIED=true)
```

### Workflow E — API request path

```text
POST /query (X-API-Key)
    → rate limit check (Redis)
    → answer cache check (Redis, keyed by question + filters + prompt_version)
    → run_agent() (same as CLI)
    → audit log write (Postgres)
    → cache store + JSON response
```

---

## 7. How to execute the full repository

### Prerequisites

- Docker + Docker Compose
- Ollama on host (`localhost:11434`)
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- ~16 GB+ VRAM for `max` profile (or CPU with `lean` + `EMBEDDING_DEVICE=cpu`)

### Step-by-step

```bash
# 1. Clone and enter repo
cd ~/path/to/biointel-agent

# 2. Pull LLM (match MODEL_PROFILE in .env)
ollama pull qwen2.5:14b-instruct

# 3. Environment
make init
# Edit .env if needed (API_KEY, OPENSEARCH_PASSWORD, NCBI_EMAIL)

# 4. Start Docker (if in dev container: sudo service docker start)
make up

# 5. DB schema
make migrate

# 6. Verify services
make healthcheck

# 7. Corpus
make seed          # fast offline demo
# OR
make ingest        # live IL-23/IBD theme from APIs

# 8. Query
make query q="What IL-23 inhibitors are used in inflammatory bowel disease?"

# 9. UI / API
# http://localhost:8501  (Streamlit)
# http://localhost:8000/docs (OpenAPI)
```

### Host vs Docker networking

- **Inside containers (API/UI):** use `localhost` for Postgres, Qdrant, Ollama (`network_mode: host`).
- **CLI on host (`make query`):** Makefile sets `HOST_ENV` with `localhost` overrides.

### Model profiles

| Profile | LLM | VRAM | Notes |
|---------|-----|------|-------|
| `max` (default) | qwen2.5:14b-instruct | ~16 GB+ | Best quality |
| `balanced` | llama3.1:8b-instruct-q4_K_M | ~8–12 GB | Good mid-range |
| `lean` | llama3.2:3b | CPU-OK | Set `EMBEDDING_DEVICE=cpu`, `RERANKER_DEVICE=cpu` |

---

## 8. How to check each service

### Automated

```bash
make healthcheck
```

Expected output pattern:

```text
[OK ] Postgres    localhost:5432/biointel
[OK ] Redis       redis://localhost:6379/0
[OK ] Qdrant      http://localhost:6333
[OK ] OpenSearch  https://localhost:9200 (cluster=green)
[OK ] MinIO       localhost:9000
[OK ] Ollama      http://localhost:11434 (model present)
```

Require Ollama for queries:

```bash
uv run python scripts/healthcheck.py --require-ollama
```

### Manual per-service checks

| Service | Check | URL / command |
|---------|-------|---------------|
| Postgres | `docker compose ps postgres` | `localhost:5432` |
| Redis | `redis-cli -h localhost ping` | `localhost:6379` |
| MinIO | Browser console | `http://localhost:9001` |
| Qdrant | `curl http://localhost:6333/readyz` | `localhost:6333` |
| OpenSearch | `curl -sk -u admin:PASSWORD https://localhost:9200/_cluster/health` | `localhost:9200` |
| API | `curl http://localhost:8000/health` | `localhost:8000/docs` |
| UI | Open browser | `http://localhost:8501` |
| Ollama | `curl http://localhost:11434/api/tags` | host only |

### Application-level

```bash
biointel info
biointel init-stores    # Qdrant collection + OS index exist
curl -s http://localhost:8000/health | jq

# API query test
curl -s http://localhost:8000/query \
  -H "X-API-Key: dev-local-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"question":"What IL-23 inhibitors are used in IBD?","top_k":8}' | jq
```

### Docker status

```bash
make ps
docker compose logs -f api
docker compose logs -f ui
```

### Observability stack (optional)

```bash
make up-obs
# Jaeger:    http://localhost:16686
# Grafana:   http://localhost:3001
# Prometheus: http://localhost:9090
# Langfuse:  http://localhost:3000
```

---

## 9. New corpus for a fresh query — how it actually works

**Interview answer:** BioIntel does **not** spin up a new Postgres database per question. It uses **one persistent multi-store corpus** that grows over time.

### Three ways to add a new topic

#### Option 1 — Auto-ingest (default for new queries)

Set in `.env`:

```env
AGENT_AUTO_INGEST=true
AGENT_AUTO_INGEST_MODE=always
AGENT_AUTO_INGEST_MAX_RECORDS=25
```

Every `biointel query` or API `/query` with `auto_ingest=true`:

1. Plans ingest from the question (e.g. `Azithromycin` for patent questions).
2. Fetches from PubMed, PMC, PatentsView, trials, ChEMBL as appropriate.
3. Persists new rows in the **same** Postgres `documents` table.
4. Indexes only **new** documents into Qdrant + OpenSearch.
5. Retrieves from the **expanded** index.

Disable for a single query:

```bash
biointel query "..." --no-auto-ingest
```

#### Option 2 — Edit static seed config (new theme, batch)

1. Edit `configs/seed_corpus.yaml` — change `topic` and per-source queries.
2. Run `make ingest` (or `biointel ingest --config ...` then `biointel index --all`).

Example for Azithromycin patents:

```yaml
topic: "Azithromycin patents"
sources:
  patentsview:
    enabled: true
    query_text: "Azithromycin"
    max_records: 100
  pubmed:
    enabled: true
    query: '"Azithromycin"[Title/Abstract] AND patent[Title/Abstract]'
```

#### Option 3 — Ad-hoc single-source ingest

```bash
biointel ingest --source patentsview --query "Azithromycin" --max-records 50
biointel ingest --source pubmed --query '"Azithromycin"[Title/Abstract]' --max-records 50
biointel index --all
```

### Full corpus reset (start from scratch)

```bash
make clean          # deletes Docker volumes (ALL data)
make up && make migrate
make seed           # or make ingest
```

Or reindex without wiping Postgres:

```bash
biointel index --reindex   # clears Qdrant + OpenSearch, re-embeds all Postgres docs
```

### Data flow for a new topic (memorize this)

```text
Official API → Connector.fetch() → normalize() → Document
    → MinIO (audit raw JSON)
    → Postgres documents table (metadata, indexed=false)
    → chunker → embedder → Qdrant (vectors) + OpenSearch (text)
    → Postgres chunks table + documents.indexed=true
    → hybrid retrieval at query time
```

### Idempotency

- Documents upsert by `doc_id` — re-ingesting the same record does not duplicate.
- Chunks use deterministic `chunk_id` — safe to re-index.
- `run_indexing(all_docs=False)` only processes documents where `indexed=false`.

---

## 10. Likely interview questions & strong answers

### Architecture & design

**Q: Why agentic RAG instead of simple RAG?**  
A: Biomedical questions are multi-faceted (trials + patents + literature). The agent **plans sub-questions**, retrieves per sub-question, **extracts structured records**, detects **contradictions**, and **verifies citations** — not just one retrieval + one LLM call.

**Q: Why hybrid retrieval?**  
A: Dense vectors capture semantics; BM25 captures exact tokens (gene symbols, NCT IDs, drug names). **RRF** fuses ranks without score normalization; cross-encoder reranker sharpens final ordering.

**Q: How do you prevent hallucinated citations?**  
A: `verify_node` uses **deterministic lexical overlap** between each cited sentence and the source chunk. If verification fails, one **regeneration** with stricter prompt. Answer carries `verified=True/False`.

**Q: Why no scraping?**  
A: Official APIs provide stable schemas, licenses, and provenance. Scraping is brittle and often violates ToS. Every `Document` stores `license` and `source_url`.

**Q: What runs locally vs in Docker?**  
A: **Ollama** on host. **Postgres, Redis, MinIO, Qdrant, OpenSearch, API, UI** in Docker. Embeddings/reranker run in API/CLI process (CPU or GPU).

### Data & storage

**Q: How is the system auditable?**  
A: Raw payloads in MinIO, metadata in Postgres, `audit_log` table stores question, chunks used, model, prompt version, citations. `prompt_version` bumps invalidate Redis answer cache.

**Q: What's in Postgres vs Qdrant vs OpenSearch?**  
A: Postgres = metadata + provenance + audit. Qdrant = embedding vectors. OpenSearch = searchable text for BM25. None alone is the "knowledge base"; they work together.

**Q: How do you scale to new drugs without redeploying?**  
A: Auto-ingest on each query (`AGENT_AUTO_INGEST_MODE=always`) or update `seed_corpus.yaml` and re-ingest. Same stores, idempotent upsert by `doc_id`.

**Q: Do you create a new database per query?**  
A: No. One Postgres DB, one Qdrant collection, one OpenSearch index. New queries **add documents** to the shared corpus via ingest + index.

### Operations

**Q: What if PatentsView is down?**  
A: Connector fails gracefully per-source; auto-ingest continues other sources; warning in answer. PubMed patent-related literature can still be indexed; full US patent claims need PatentsView or optional Google Patents BigQuery.

**Q: What if Ollama is down?**  
A: Ingest and index still work. Queries fail because the agent needs the LLM for plan, extract, synthesize, and contradiction steps.

**Q: How do you run without a GPU?**  
A: `MODEL_PROFILE=lean`, `EMBEDDING_DEVICE=cpu`, `RERANKER_DEVICE=cpu`. Pipeline is identical; slower embedding and LLM.

### Models & config

**Q: What models are used?**  
A: Default `max` profile: `qwen2.5:14b-instruct` (Ollama), `BAAI/bge-large-en-v1.5` (embeddings, 1024-d), `BAAI/bge-reranker-v2-m3` (reranker). Profiles in `configs/models.yaml`.

**Q: What happens if you change the embedding model?**  
A: Vector dimension may change → must `biointel index --reindex` because Qdrant collection dimension must match.

### Testing

**Q: How is the project tested without GPU/network?**  
A: Offline pytest with fixtures (`tests/fixtures/`), fake deps in agent tests, connector normalize tests, RRF math unit tests, citation verification tests. CI runs `pytest -m "not integration"`.

---

## 11. Quick reference cheat sheet

### Lifecycle commands

```bash
make init → make up → make migrate → make healthcheck → make seed → make query
```

### New topic

```bash
# Automatic (default)
make query q="What patents claim Azithromycin?"

# Manual
biointel ingest --source patentsview --query "Azithromycin" --max-records 50
biointel index --all
```

### Reset

```bash
make clean && make up && make migrate && make seed
biointel index --reindex    # rebuild vectors only
```

### Debug

```bash
biointel info
make ps
make logs
docker compose logs -f api
```

### URLs

| URL | Service |
|-----|---------|
| http://localhost:8000/docs | API (OpenAPI) |
| http://localhost:8501 | Streamlit UI |
| http://localhost:9001 | MinIO console |
| http://localhost:6333/dashboard | Qdrant |
| http://localhost:16686 | Jaeger (obs profile) |
| http://localhost:3001 | Grafana (obs profile) |
| http://localhost:3000 | Langfuse (obs profile) |

### Key environment variables

| Variable | Purpose |
|----------|---------|
| `MODEL_PROFILE` | `max` / `balanced` / `lean` |
| `API_KEY` | Required in `X-API-Key` header |
| `AGENT_AUTO_INGEST` | Fetch sources before each query |
| `AGENT_AUTO_INGEST_MODE` | `always` or `if_needed` |
| `EMBEDDING_DEVICE` / `RERANKER_DEVICE` | `cuda` or `cpu` |
| `VECTOR_BACKEND` | `qdrant` or `milvus` |
| `PATENTSVIEW_API_KEY` | Optional USPTO PatentSearch API key |
| `NCBI_EMAIL` / `NCBI_API_KEY` | PubMed etiquette / higher rate limits |

---

## 5-minute verbal pitch

> "I built BioIntel Agent — a local agentic RAG system for drug discovery and patent intelligence. It ingests from official APIs like PubMed, ClinicalTrials.gov, ChEMBL, and USPTO PatentsView — no scraping — and stores metadata in Postgres, raw payloads in MinIO, vectors in Qdrant, and BM25 text in OpenSearch.
>
> When you ask a question, the LangGraph agent decomposes it into sub-questions, runs hybrid retrieval with RRF fusion and cross-encoder reranking, extracts structured records per document type, synthesizes a cited answer, flags contradictions across sources, and verifies every citation with a deterministic lexical check — not just the LLM saying it's correct.
>
> For new topics like a different drug, auto-ingest fetches and indexes relevant documents on each query into the shared corpus. Everything runs locally with Ollama and open-weight models — no data leaves the host. The API adds auth, rate limiting, caching, and a full audit log."

---

*See also: [README.md](README.md) for user-facing docs and [configs/models.yaml](configs/models.yaml) for model profiles.*
