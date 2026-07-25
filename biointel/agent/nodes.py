"""Agent node functions: plan, retrieve, extract, synthesize, contradictions, verify.

Each node is a plain function ``(state, deps) -> partial_state`` so it can be unit
tested in isolation with fake dependencies (no LLM, no stores). ``graph.py``
wires them into a LangGraph ``StateGraph``; the same functions power the pure
fallback runner, guaranteeing identical behavior with or without LangGraph.

``deps`` is a small container holding the LLM client and hybrid retriever so tests
can inject fakes and production can inject the real, lazily-constructed services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biointel.agent import prompts
from biointel.agent.state import AgentState
from biointel.agent.structures import (
    EXTRACTION_SCHEMAS,
    ContradictionReport,
    QueryPlan,
)
from biointel.agent.verify import verify_answer
from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import Contradiction, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class AgentDeps:
    """Injected dependencies so nodes stay testable and decoupled."""

    llm: Any
    retriever: Any

    @classmethod
    def build(cls) -> AgentDeps:
        from biointel.agent.llm import get_llm
        from biointel.retrieval.hybrid import get_hybrid_retriever

        return cls(llm=get_llm(), retriever=get_hybrid_retriever())


# --------------------------------------------------------------------------
# 1. PLAN
# --------------------------------------------------------------------------
def plan_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Decompose the question into retrievable sub-questions (LLM, JSON-constrained)."""
    question = state["question"]
    instructions = prompts.PLANNER_INSTRUCTIONS.format(
        max_sub_questions=settings.agent_max_sub_questions
    )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"QUESTION:\n{question}"},
    ]
    try:
        plan: QueryPlan = deps.llm.generate_json(messages, QueryPlan)
        subs = [sq.question.strip() for sq in plan.sub_questions if sq.question.strip()]
        raw = [sq.model_dump() for sq in plan.sub_questions]
    except Exception as exc:  # pragma: no cover - LLM/runtime dependent
        logger.warning("Planning failed (%s); falling back to single-question plan.", exc)
        subs, raw = [], []

    if not subs:
        subs = [question]
        raw = [{"question": question, "source_type": None}]
    subs = subs[: settings.agent_max_sub_questions]
    logger.info("Planned %d sub-question(s).", len(subs))
    return {"sub_questions": subs, "plan_raw": raw}


# --------------------------------------------------------------------------
# 2. RETRIEVE
# --------------------------------------------------------------------------
def retrieve_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Hybrid-retrieve for each sub-question, then merge + dedupe by chunk_id."""
    base_filters = state.get("filters") or {}
    per_q_k = state.get("top_k") or settings.retrieval_final_top_k
    plan_raw = state.get("plan_raw") or [{"question": q} for q in state["sub_questions"]]

    merged: dict[str, RetrievedChunk] = {}
    for item in plan_raw:
        q = item["question"]
        filters = dict(base_filters)
        st = item.get("source_type")
        if st and "source" not in filters:
            filters["source"] = st
        hits = deps.retriever.retrieve(q, top_k=per_q_k, filters=filters or None)
        for rc in hits:
            cid = rc.chunk.chunk_id
            # Keep the best-scoring instance of a chunk seen across sub-questions.
            prev = merged.get(cid)
            if prev is None or (rc.rerank_score or rc.fused_score or 0) > (
                prev.rerank_score or prev.fused_score or 0
            ):
                merged[cid] = rc

    ranked = sorted(
        merged.values(),
        key=lambda c: c.rerank_score if c.rerank_score is not None else (c.fused_score or 0),
        reverse=True,
    )
    ranked = ranked[: settings.agent_context_max_chunks]
    required = [t.lower() for t in (state.get("required_terms") or []) if t]
    if required:
        filtered = [
            rc
            for rc in ranked
            if any(
                term in rc.chunk.text.lower() or term in (rc.chunk.title or "").lower()
                for term in required
            )
        ]
        if filtered:
            ranked = filtered
        else:
            logger.info(
                "No chunks matched required terms %s; dropping irrelevant hits.",
                required[:5],
            )
            ranked = []
    for i, rc in enumerate(ranked):
        rc.rank = i + 1
    logger.info("Retrieved %d unique context chunks.", len(ranked))
    return {"retrieved": ranked}


# --------------------------------------------------------------------------
# 3. EXTRACT (structured records per chunk)
# --------------------------------------------------------------------------
def extract_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Turn top chunks into typed records based on their doc_type.

    Extraction is best-effort and additive: failures are logged and skipped so a
    single bad extraction never blocks the answer. Records make the agent's
    reasoning auditable and feed comparable fields for contradiction analysis.
    """
    retrieved = state.get("retrieved", [])
    records: list[dict[str, Any]] = []
    # Bound extraction cost: only the strongest few chunks.
    for rc in retrieved[:6]:
        dt = rc.chunk.doc_type.value
        schema = EXTRACTION_SCHEMAS.get(dt)
        if schema is None:
            continue
        instr = prompts.EXTRACTION_INSTRUCTIONS.get(dt, "Extract structured fields.")
        messages = [
            {"role": "system", "content": instr},
            {"role": "user", "content": f"SOURCE:\n{rc.chunk.text}"},
        ]
        try:
            rec = deps.llm.generate_json(messages, schema)
            records.append(
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "doc_id": rc.chunk.doc_id,
                    "doc_type": dt,
                    "record": rec.model_dump(exclude_none=True),
                }
            )
        except Exception as exc:  # pragma: no cover - LLM dependent
            logger.debug("Extraction skipped for %s (%s)", rc.chunk.chunk_id, exc)
    logger.info("Extracted %d structured record(s).", len(records))
    return {"structured": records}


# --------------------------------------------------------------------------
# 4. SYNTHESIZE
# --------------------------------------------------------------------------
def synthesize_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Generate a cited answer from the retrieved SOURCES block."""
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {
            "answer": "I could not find any relevant sources in the knowledge base to "
            "answer this question.",
            "model": deps.llm.model,
            "prompt_version": settings.prompt_version,
        }
    block = prompts.sources_block(retrieved)
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.synthesis_prompt(state["question"], block)},
    ]
    answer = deps.llm.chat(messages)
    return {
        "answer": answer,
        "model": deps.llm.model,
        "prompt_version": settings.prompt_version,
    }


# --------------------------------------------------------------------------
# 5. CONTRADICTION CHECK
# --------------------------------------------------------------------------
def contradiction_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Detect cross-source contradictions among the retrieved SOURCES (LLM, JSON)."""
    retrieved = state.get("retrieved", [])
    if len(retrieved) < 2:
        return {"contradictions": []}
    block = prompts.sources_block(retrieved)
    messages = [
        {"role": "system", "content": prompts.CONTRADICTION_INSTRUCTIONS},
        {"role": "user", "content": f"SOURCES:\n{block}"},
    ]
    contradictions: list[Contradiction] = []
    try:
        report: ContradictionReport = deps.llm.generate_json(messages, ContradictionReport)
        for item in report.contradictions:
            contradictions.append(
                Contradiction(
                    statement_a=item.statement_a,
                    statement_b=item.statement_b,
                    source_a=_marker_to_label(item.source_a, retrieved),
                    source_b=_marker_to_label(item.source_b, retrieved),
                    explanation=item.explanation,
                )
            )
    except Exception as exc:  # pragma: no cover - LLM dependent
        logger.warning("Contradiction detection failed (%s); returning none.", exc)
    logger.info("Detected %d contradiction(s).", len(contradictions))
    return {"contradictions": contradictions}


def _marker_to_label(marker: str, retrieved: list[RetrievedChunk]) -> str:
    """Map a source number like '3' back to a human-readable provenance label."""
    digits = "".join(ch for ch in str(marker) if ch.isdigit())
    if digits:
        idx = int(digits)
        if 1 <= idx <= len(retrieved):
            return retrieved[idx - 1].chunk.citation_label()
    return str(marker)


# --------------------------------------------------------------------------
# 6. VERIFY (+ decide whether to regenerate)
# --------------------------------------------------------------------------
def verify_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Run citation verification; mark verified + collect citations/warnings."""
    retrieved = state.get("retrieved", [])
    answer = state.get("answer", "")
    result = verify_answer(answer, retrieved)
    return {
        "citations": result.citations,
        "verified": result.verified,
        "warnings": list(result.warnings),
    }


def should_regenerate(state: AgentState) -> bool:
    """Conditional edge predicate: regenerate once if unverified and allowed."""
    return (
        settings.agent_regenerate_on_unverified
        and not state.get("verified", False)
        and not state.get("regenerated", False)
        and bool(state.get("retrieved"))
    )


def regenerate_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Stricter re-synthesis pass that drops unsupported claims.

    We feed back the fact that the previous answer had unsupported sentences and
    instruct the model to only make claims it can cite. Marked ``regenerated`` so
    the graph does not loop.
    """
    retrieved = state.get("retrieved", [])
    block = prompts.sources_block(retrieved)
    stricter = (
        prompts.synthesis_prompt(state["question"], block)
        + "\n\nIMPORTANT: A previous draft contained sentences that were not supported "
        "by the sources. Rewrite so that EVERY factual sentence ends with a valid [n] "
        "citation to a source that actually states it. Omit any claim you cannot cite."
    )
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": stricter},
    ]
    answer = deps.llm.chat(messages, temperature=0.0)
    return {"answer": answer, "regenerated": True}
