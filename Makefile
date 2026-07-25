# =============================================================================
# BioIntel Agent — developer command surface.
# Run `make help` for the list.
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

# Env overrides for CLI/tools run on the host against docker-compose port mappings.
# Containers use docker DNS names from .env; the host must use localhost.
HOST_ENV := \
	POSTGRES_HOST=localhost \
	REDIS_URL=redis://localhost:6379/0 \
	MINIO_ENDPOINT=localhost:9000 \
	QDRANT_URL=http://localhost:6333 \
	OPENSEARCH_URL=https://localhost:9200 \
	OLLAMA_BASE_URL=http://localhost:11434

# Host venv PyTorch may target a newer CUDA than the workstation driver supports.
# Override when GPU works: make seed HOST_ML_ENV="EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda"
HOST_ML_ENV ?= EMBEDDING_DEVICE=cpu RERANKER_DEVICE=cpu
HOST_RUN := $(HOST_ENV) $(HOST_ML_ENV)

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment
.PHONY: init
init: ## Create .env from template and install dev deps into a local venv (uv)
	@test -f .env || cp .env.example .env
	@test -d .venv || uv venv
	uv pip install -e ".[dev,milvus,gcp]" --python .venv
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		uv run pre-commit install; \
	else \
		echo "Skipping pre-commit install (not a git repository)"; \
	fi
	@echo "✓ Environment ready. Edit .env if needed, then 'make up'."

# --------------------------------------------------------------------- docker
.PHONY: build
build: ## Build the API/UI image
	$(COMPOSE) build

.PHONY: up
up: ## Start the base stack (no observability) in the background
	$(COMPOSE) up -d
	@echo "✓ Base stack up. API: http://localhost:8000/docs  UI: http://localhost:8501"

.PHONY: up-obs
up-obs: ## Start stack WITH observability (Langfuse, Jaeger, Prometheus, Grafana)
	$(COMPOSE) --profile observability up -d
	@echo "✓ Stack + observability up. Jaeger:16686 Grafana:3001 Langfuse:3000"

.PHONY: down
down: ## Stop all services (keep volumes)
	$(COMPOSE) --profile observability --profile milvus down

.PHONY: clean
clean: ## Stop services AND delete all data volumes (destructive)
	$(COMPOSE) --profile observability --profile milvus down -v

.PHONY: logs
logs: ## Tail API logs
	$(COMPOSE) logs -f api

.PHONY: ps
ps: ## Show service status
	$(COMPOSE) ps

# ------------------------------------------------------------------ migrations
.PHONY: migrate
migrate: ## Apply DB migrations (alembic upgrade head)
	$(HOST_RUN) uv run alembic upgrade head

.PHONY: revision
revision: ## Autogenerate a new migration: make revision m="message"
	$(HOST_RUN) uv run alembic revision --autogenerate -m "$(m)"

# --------------------------------------------------------------------- ingest
.PHONY: seed
seed: ## Ingest + index the small OFFLINE fixture corpus (no network) for a quick demo
	$(HOST_RUN) uv run python scripts/seed_demo.py

.PHONY: ingest
ingest: ## Live ingest using configs/seed_corpus.yaml (hits official APIs)
	$(HOST_RUN) uv run biointel ingest --config configs/seed_corpus.yaml; \
	$(HOST_RUN) uv run biointel index --all

.PHONY: query
query: ## Run a demo query: make query q="your question"
ifndef q
	$(error Pass the question with q=, e.g. make query q="What IL-23 inhibitors are used in IBD?")
endif
	$(HOST_RUN) uv run biointel query "$(q)"

# ------------------------------------------------------------------------ run
.PHONY: api
api: ## Run the API locally (without docker)
	$(HOST_RUN) uv run uvicorn biointel.api.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: ui
ui: ## Run the Streamlit UI locally (without docker)
	$(HOST_RUN) uv run streamlit run biointel/ui/app.py

# ---------------------------------------------------------------------- checks
.PHONY: test
test: ## Run the offline test suite
	uv run pytest -m "not integration"

.PHONY: test-all
test-all: ## Run ALL tests including integration (requires services up)
	uv run pytest

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check biointel tests scripts

.PHONY: fmt
fmt: ## Auto-format with ruff
	uv run ruff format biointel tests scripts
	uv run ruff check --fix biointel tests scripts

.PHONY: healthcheck
healthcheck: ## Verify all services are reachable
	$(HOST_RUN) uv run python scripts/healthcheck.py

# ------------------------------------------------------------------------- zip
.PHONY: zip
zip: ## Package the repo into a distributable .zip
	uv run python scripts/package_zip.py
