"""Shared state object passed between LangGraph nodes.

Using a ``TypedDict`` keeps the graph state explicit and serializable. Each node
reads the fields it needs and returns a partial dict that LangGraph merges into
the running state, so the flow of data through plan -> retrieve -> extract ->
synthesize -> contradictions -> verify is easy to follow and to test.
"""

from __future__ import annotations

from typing import Any, TypedDict

from biointel.common.schemas import Citation, Contradiction, RetrievedChunk


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    top_k: int
    filters: dict[str, Any]
    required_terms: list[str]  # chunks must mention at least one term (post-ingest)
    trace_id: str

    # planning
    sub_questions: list[str]
    plan_raw: list[dict[str, Any]]  # [{question, source_type}]

    # retrieval
    retrieved: list[RetrievedChunk]  # final, deduped, reranked context set

    # extraction
    structured: list[dict[str, Any]]  # typed records per chunk/doc_type

    # synthesis + verification
    answer: str
    citations: list[Citation]
    contradictions: list[Contradiction]
    verified: bool
    regenerated: bool
    warnings: list[str]

    # bookkeeping
    model: str
    prompt_version: str
