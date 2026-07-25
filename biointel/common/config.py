"""Centralized, typed configuration loaded from environment / .env.

All subsystems import ``settings`` from here. Model *profiles* (max / balanced /
lean) provide coherent defaults; individual env vars always override the profile.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Model profiles: coherent (LLM, embedding, reranker) triples.
# A profile only sets defaults; any explicit env var wins over the profile.
# ---------------------------------------------------------------------------
MODEL_PROFILES: dict[str, dict[str, object]] = {
    "max": {
        "llm_model": "qwen2.5:14b-instruct",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
    },
    "balanced": {
        "llm_model": "llama3.1:8b-instruct-q4_K_M",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
    },
    "lean": {
        "llm_model": "llama3.2:3b",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "reranker_model": "BAAI/bge-reranker-base",
    },
}


class Settings(BaseSettings):
    """Application settings. Values come from env vars / .env (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- General ----
    biointel_env: str = "local"
    log_level: str = "INFO"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "dev-local-key-change-me"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    api_cors_origins: str = "*"  # comma-separated; "*" allows all (dev default)
    answer_cache_enabled: bool = True
    answer_cache_ttl_seconds: int = 3600

    # ---- Model profile ----
    model_profile: Literal["max", "balanced", "lean"] = "max"

    # ---- LLM (Ollama) ----
    ollama_base_url: str = "http://host.docker.internal:11434"
    llm_model: str | None = None  # resolved from profile if unset
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout_seconds: int = 120

    # ---- Agent ----
    # Bumping prompt_version invalidates cached answers and is recorded in the
    # audit log so every stored answer is traceable to the exact prompt logic.
    prompt_version: str = "v1"
    agent_max_sub_questions: int = 4
    agent_context_max_chunks: int = 12  # chunks passed to the synthesis LLM
    agent_regenerate_on_unverified: bool = True  # one retry if citations fail
    citation_min_overlap: float = 0.30  # token-overlap floor for sentence support
    agent_auto_ingest: bool = True  # fetch + index sources for the question before retrieve
    agent_auto_ingest_mode: Literal["always", "if_needed"] = "always"
    agent_auto_ingest_max_records: int = 25  # per source during auto-ingest

    # ---- Embeddings ----
    embedding_model: str | None = None  # resolved from profile if unset
    embedding_dim: int | None = None  # resolved from profile if unset
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages:"

    # ---- Reranker ----
    reranker_model: str | None = None  # resolved from profile if unset
    reranker_device: str = "cuda"
    reranker_top_k: int = 8
    reranker_max_length: int = 512

    # ---- Vector / search ----
    vector_backend: Literal["qdrant", "milvus"] = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "biointel_chunks"
    milvus_uri: str = "http://milvus:19530"
    milvus_collection: str = "biointel_chunks"

    opensearch_url: str = "https://opensearch:9200"
    opensearch_user: str = "admin"
    opensearch_password: str = "BioIntel_Admin_123!"
    opensearch_index: str = "biointel_chunks"
    opensearch_verify_certs: bool = False

    retrieval_dense_top_k: int = 25
    retrieval_sparse_top_k: int = 25
    rrf_k: int = 60
    retrieval_final_top_k: int = 8

    # ---- Infra: Postgres ----
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "biointel"
    postgres_user: str = "biointel"
    postgres_password: str = "biointel"
    database_url: str | None = None  # derived if unset

    # ---- Infra: Redis ----
    redis_url: str = "redis://redis:6379/0"

    # ---- Infra: MinIO ----
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "biointel-raw"
    minio_secure: bool = False

    # ---- Observability ----
    obs_enabled: bool = False
    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://jaeger:4317"
    otel_service_name: str = "biointel-agent"

    # ---- Ingestion credentials (optional) ----
    ncbi_api_key: str = ""
    ncbi_email: str = "you@example.com"
    ncbi_tool: str = "biointel-agent"
    patentsview_api_key: str = ""  # USPTO PatentSearch API key (account.uspto.gov/api-manager)
    google_application_credentials: str = ""
    gcp_project_id: str = ""

    # ------------------------------------------------------------------
    # Profile resolution: fill any unset model fields from the profile.
    # ------------------------------------------------------------------
    def model_post_init(self, __context: object) -> None:  # noqa: D401
        profile = MODEL_PROFILES[self.model_profile]
        if self.llm_model is None:
            object.__setattr__(self, "llm_model", profile["llm_model"])
        if self.embedding_model is None:
            object.__setattr__(self, "embedding_model", profile["embedding_model"])
        if self.embedding_dim is None:
            object.__setattr__(self, "embedding_dim", profile["embedding_dim"])
        if self.reranker_model is None:
            object.__setattr__(self, "reranker_model", profile["reranker_model"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_database_url(self) -> str:
        """SQLAlchemy URL, derived from parts unless explicitly overridden."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def google_patents_enabled(self) -> bool:
        return bool(self.google_application_credentials and self.gcp_project_id)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor (import this everywhere)."""
    return Settings()


settings = get_settings()
