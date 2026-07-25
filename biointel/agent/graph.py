"""LangGraph assembly + the public ``run_agent`` entry point.

The agent is a directed graph:

    plan -> retrieve -> extract -> synthesize -> contradictions -> verify
                                                        ^                |
                                                        |   (unverified) |
                                                        +--- regenerate <+
                                                                         |
                                                                    (verified) -> END

``verify`` uses a conditional edge: if the answer failed citation verification and
regeneration is enabled, it routes to ``regenerate`` (a stricter re-synthesis)
which loops back through ``verify`` exactly once (guarded by the ``regenerated``
flag), otherwise it ends.

If LangGraph is not importable (e.g. a lean environment), :func:`run_agent`
transparently falls back to :func:`_run_linear`, which executes the very same node
functions in the same order. This guarantees identical behavior and keeps the
whole pipeline unit-testable offline without the LangGraph dependency.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from biointel.agent.nodes import (
    AgentDeps,
    contradiction_node,
    extract_node,
    plan_node,
    regenerate_node,
    retrieve_node,
    should_regenerate,
    synthesize_node,
    verify_node,
)
from biointel.agent.state import AgentState
from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import AgentAnswer, SourceType

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# LangGraph build
# --------------------------------------------------------------------------
def build_graph(deps: AgentDeps):
    """Build and compile the LangGraph ``StateGraph``. Raises if LangGraph absent."""
    from langgraph.graph import END, StateGraph

    def _wrap(fn):
        # Bind deps and adapt (state)->partial-state signature for LangGraph.
        def node(state: AgentState) -> dict[str, Any]:
            return fn(state, deps)

        return node

    g = StateGraph(AgentState)
    g.add_node("plan", _wrap(plan_node))
    g.add_node("retrieve", _wrap(retrieve_node))
    g.add_node("extract", _wrap(extract_node))
    g.add_node("synthesize", _wrap(synthesize_node))
    g.add_node("contradictions", _wrap(contradiction_node))
    g.add_node("verify", _wrap(verify_node))
    g.add_node("regenerate", _wrap(regenerate_node))

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "synthesize")
    g.add_edge("synthesize", "contradictions")
    g.add_edge("contradictions", "verify")

    def _route(state: AgentState) -> str:
        return "regenerate" if should_regenerate(state) else END

    g.add_conditional_edges("verify", _route, {"regenerate": "regenerate", END: END})
    g.add_edge("regenerate", "verify")
    return g.compile()


# --------------------------------------------------------------------------
# Pure fallback runner (identical order, no LangGraph dependency)
# --------------------------------------------------------------------------
def _run_linear(state: AgentState, deps: AgentDeps) -> AgentState:
    state.update(plan_node(state, deps))
    state.update(retrieve_node(state, deps))
    state.update(extract_node(state, deps))
    state.update(synthesize_node(state, deps))
    state.update(contradiction_node(state, deps))
    state.update(verify_node(state, deps))
    if should_regenerate(state):
        state.update(regenerate_node(state, deps))
        state.update(verify_node(state, deps))
    return state


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def run_agent(
    question: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    deps: AgentDeps | None = None,
    trace_id: str | None = None,
    auto_ingest: bool | None = None,
) -> AgentAnswer:
    """Run the full agent for one question and return a structured answer.

    Parameters
    ----------
    question: user question.
    top_k: chunks retrieved per sub-question (defaults to config).
    filters: optional metadata filters (source/doc_type/date) applied to retrieval.
    deps: injected dependencies (tests pass fakes); built from settings if None.
    trace_id: correlation id for observability/audit; generated if None.
    """
    deps = deps or AgentDeps.build()
    trace_id = trace_id or uuid.uuid4().hex
    do_ingest = settings.agent_auto_ingest if auto_ingest is None else auto_ingest
    bootstrap_warnings: list[str] = []

    retrieval_filters = dict(filters or {})
    required_terms: list[str] = []

    if do_ingest:
        from biointel.ingestion.dynamic import (
            ensure_corpus_for_question,
            infer_retrieval_context,
        )

        stats = ensure_corpus_for_question(
            question,
            llm=deps.llm,
            retriever=deps.retriever,
        )
        ctx = infer_retrieval_context(question, stats)
        retrieval_filters.update(ctx.filters)
        required_terms = ctx.required_terms

        if stats.ingested:
            bootstrap_warnings.append(
                f"Auto-ingested {stats.ingested} document(s) from "
                f"{', '.join(stats.sources_run)} "
                f"({stats.indexed_chunks} chunks indexed) for topic: {stats.topic[:80]}."
            )
            logger.info(
                "Auto-ingest complete: %d docs, %d chunks", stats.ingested, stats.indexed_chunks
            )
        if stats.is_patent_question and SourceType.PATENTSVIEW.value in stats.failed_sources:
            bootstrap_warnings.append(
                "USPTO PatentsView API was unreachable; patent claims could not be fetched. "
                "Set PATENTSVIEW_API_KEY and ensure search.patentsview.org is reachable."
            )

    state: AgentState = {
        "question": question,
        "top_k": top_k or settings.retrieval_final_top_k,
        "filters": retrieval_filters,
        "required_terms": required_terms,
        "trace_id": trace_id,
        "regenerated": False,
    }

    started = time.perf_counter()
    try:
        app = build_graph(deps)
        final: AgentState = app.invoke(state)
        engine = "langgraph"
    except ImportError:
        logger.info("LangGraph not available; using linear fallback runner.")
        final = _run_linear(state, deps)
        engine = "linear"
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Agent finished via %s in %.0f ms (verified=%s)", engine, elapsed_ms, final.get("verified")
    )

    return AgentAnswer(
        query=question,
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        contradictions=final.get("contradictions", []),
        used_chunks=[rc.chunk.chunk_id for rc in final.get("retrieved", [])],
        sub_questions=final.get("sub_questions", []),
        model=final.get("model", deps.llm.model),
        verified=final.get("verified", False),
        warnings=bootstrap_warnings + list(final.get("warnings", [])),
    )
