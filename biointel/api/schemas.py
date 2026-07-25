"""Request/response models for the HTTP API (the public contract).

These are separate from the internal domain models in ``common.schemas`` so the
wire format can evolve independently of the agent's internals. ``AgentAnswer`` is
reused directly in :class:`QueryResponse` because it is already a clean, typed
representation of a verified answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from biointel.common.schemas import AgentAnswer


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional retrieval filters, e.g. {'source': 'pubmed'} or "
        "{'doc_type': ['trial','paper'], 'date_from': '2020-01-01'}.",
    )
    session_id: str | None = Field(default=None, description="Chat session to append to.")
    use_cache: bool = Field(default=True, description="Allow returning a cached answer.")
    auto_ingest: bool | None = Field(
        default=None,
        description="Fetch + index live sources for the question before answering. "
        "Defaults to AGENT_AUTO_INGEST from settings.",
    )


class QueryResponse(BaseModel):
    answer: AgentAnswer
    trace_id: str
    session_id: str | None = None
    latency_ms: int
    cached: bool = False


class IngestRequest(BaseModel):
    source: str = Field(description="Connector id, e.g. 'clinicaltrials'.")
    query: str | None = Field(default=None, description="Search term for the connector.")
    max_records: int = Field(default=25, ge=1, le=500)
    index: bool = Field(default=True, description="Index documents after ingesting.")


class IngestResponse(BaseModel):
    source: str
    ingested: int
    indexed: int
    message: str


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    doc_type: str
    title: str
    license: str
    indexed: bool
    n_chunks: int


class DocumentsResponse(BaseModel):
    total: int
    documents: list[DocumentSummary]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_profile: str
    llm_model: str
    embedding_model: str
    components: dict[str, str]
