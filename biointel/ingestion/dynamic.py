"""Query-driven corpus bootstrap: ingest + index sources matched to a question.

When ``agent_auto_ingest`` is enabled, the agent can fetch fresh documents from
official APIs for the user's topic before retrieval, so questions like "kRAS
patents" are not limited to a pre-seeded IL-23 demo corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from biointel.agent.structures import DynamicIngestPlan
from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import SourceType
from biointel.indexing.runner import run_indexing
from biointel.ingestion.runner import ingest_single

logger = get_logger(__name__)

_GENE_RE = re.compile(r"\b(?:[A-Z][A-Z0-9-]{1,9}|IL-\d+[A-Z]?)\b")
_DRUG_RE = re.compile(
    r"\b[A-Z][a-z]{3,}(?:micin|mab|nib|zumab|vir|cillin|azole|stat|pril|olol|cycline)?\b"
)
_PATENT_HINTS = ("patent", "claim", "intellectual property", "ip ")
_TRIAL_HINTS = ("trial", "clinical", "phase ", "nct", "recruit")
_QUESTION_STOPWORDS = frozenset(
    {
        "what",
        "which",
        "patents",
        "patent",
        "claim",
        "claims",
        "about",
        "the",
        "a",
        "an",
        "do",
        "does",
        "include",
        "are",
        "is",
        "for",
        "in",
        "on",
        "with",
        "how",
        "many",
    }
)


@dataclass
class IngestStats:
    ingested: int = 0
    indexed_chunks: int = 0
    sources_run: list[str] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    topic: str = ""
    primary_term: str = ""
    is_patent_question: bool = False


@dataclass
class RetrievalContext:
    """Filters and term gates applied after auto-ingest."""

    filters: dict[str, Any] = field(default_factory=dict)
    required_terms: list[str] = field(default_factory=list)


def is_patent_question(question: str) -> bool:
    lower = question.lower()
    return any(h in lower for h in _PATENT_HINTS)


def extract_primary_term(question: str) -> str | None:
    """Pull the main drug/gene/compound from a natural-language question."""
    genes = _GENE_RE.findall(question)
    if genes:
        return genes[0]

    drugs = [d for d in _DRUG_RE.findall(question) if d.lower() not in _QUESTION_STOPWORDS]
    if drugs:
        return max(drugs, key=len)

    words = [
        w
        for w in re.findall(r"[A-Za-z0-9-]{3,}", question)
        if w.lower() not in _QUESTION_STOPWORDS
    ]
    return words[-1] if words else None


def heuristic_ingest_plan(question: str) -> DynamicIngestPlan:
    """Build a conservative ingest plan without calling the LLM."""
    q = question.strip()
    lower = q.lower()
    primary = extract_primary_term(q)
    genes = list(dict.fromkeys(_GENE_RE.findall(q)))[:5]
    wants_patents = is_patent_question(q)
    wants_trials = any(h in lower for h in _TRIAL_HINTS)
    topic = primary or q[:80]

    if wants_patents and primary:
        patent_query = primary
        pubmed_query = f'"{primary}"[Title/Abstract] AND patent[Title/Abstract]'
        pmc_query = f'("{primary}"[Title/Abstract] AND patent[Title/Abstract]) AND open access[filter]'
    elif primary:
        patent_query = f"{primary} patent"
        pubmed_query = f'"{primary}"[Title/Abstract]'
        pmc_query = f'"{primary}"[Title/Abstract] AND open access[filter]'
    else:
        patent_query = q if wants_patents else f"{q} patent"
        pubmed_query = q
        pmc_query = f"{q} AND open access[filter]" if "open access" not in lower else q

    return DynamicIngestPlan(
        topic=topic,
        pubmed_query=pubmed_query,
        pmc_query=pmc_query,
        patent_query=patent_query if wants_patents or primary else patent_query,
        trial_conditions=[topic[:120]] if wants_trials else [],
        trial_terms=primary or q if wants_trials else None,
        gene_targets=genes,
        include_literature=True,
        include_patents=wants_patents or bool(primary),
        include_trials=wants_trials or bool(genes),
        include_compounds=bool(genes),
    )


def plan_ingest_for_question(question: str, llm: Any | None = None) -> DynamicIngestPlan:
    """LLM-authored ingest plan with a deterministic heuristic fallback."""
    if llm is None:
        return heuristic_ingest_plan(question)

    from biointel.agent import prompts

    messages = [
        {"role": "system", "content": prompts.INGEST_PLANNER_INSTRUCTIONS},
        {"role": "user", "content": f"QUESTION:\n{question.strip()}"},
    ]
    try:
        plan: DynamicIngestPlan = llm.generate_json(messages, DynamicIngestPlan)
        if not plan.topic:
            plan.topic = extract_primary_term(question) or question[:160]
        primary = extract_primary_term(question)
        if primary and is_patent_question(question):
            plan.patent_query = primary
        return plan
    except Exception as exc:  # pragma: no cover - LLM dependent
        logger.warning("Dynamic ingest planning failed (%s); using heuristics.", exc)
        return heuristic_ingest_plan(question)


def question_terms(question: str) -> list[str]:
    """Terms that retrieved chunks should mention to be considered on-topic."""
    terms: list[str] = []
    primary = extract_primary_term(question)
    if primary:
        terms.append(primary)
    terms.extend(_GENE_RE.findall(question))
    for word in re.findall(r"[A-Za-z0-9-]{4,}", question):
        if word.lower() not in _QUESTION_STOPWORDS:
            terms.append(word)
    return list(dict.fromkeys(t.lower() for t in terms if t))


def infer_retrieval_context(question: str, stats: IngestStats) -> RetrievalContext:
    """Derive retrieval filters after auto-ingest based on question intent."""
    terms = [t.lower() for t in question_terms(question)]
    ctx = RetrievalContext(required_terms=terms)
    ctx.filters = {}

    if stats.is_patent_question and SourceType.PATENTSVIEW.value in stats.sources_run:
        ctx.filters["doc_type"] = "patent"

    return ctx


def _needs_ingest(question: str, retriever: Any) -> bool:
    """Return True when the indexed corpus is unlikely to answer the question."""
    try:
        hits = retriever.retrieve(question, top_k=5)
    except Exception as exc:  # pragma: no cover
        logger.warning("Retrieval probe failed (%s); will ingest.", exc)
        return True
    if len(hits) < 2:
        return True

    terms = {t.lower() for t in question_terms(question)}
    if terms:
        top_text = " ".join(h.chunk.text.lower() for h in hits[:3])
        if not any(term in top_text for term in terms):
            logger.info("Top hits do not mention query terms %s; will ingest.", sorted(terms)[:5])
            return True

    best = max((h.rerank_score or h.fused_score or 0.0) for h in hits)
    return best < 0.02


def ensure_corpus_for_question(
    question: str,
    *,
    llm: Any | None = None,
    retriever: Any | None = None,
    mode: Literal["always", "if_needed"] | None = None,
    max_records: int | None = None,
) -> IngestStats:
    """Ingest + index documents tailored to ``question``."""
    mode = mode or settings.agent_auto_ingest_mode
    cap = max_records or settings.agent_auto_ingest_max_records
    primary = extract_primary_term(question) or ""
    stats = IngestStats(
        topic=primary or question[:160],
        primary_term=primary,
        is_patent_question=is_patent_question(question),
    )

    if mode == "if_needed":
        if retriever is None:
            from biointel.retrieval.hybrid import get_hybrid_retriever

            retriever = get_hybrid_retriever()
        if not _needs_ingest(question, retriever):
            logger.info("Corpus looks sufficient for this question; skipping auto-ingest.")
            return stats

    plan = plan_ingest_for_question(question, llm=llm)
    stats.topic = plan.topic
    logger.info("Auto-ingest topic: %s", plan.topic)

    literature_jobs: list[tuple[str, dict[str, Any]]] = []
    patent_jobs: list[tuple[str, dict[str, Any]]] = []
    other_jobs: list[tuple[str, dict[str, Any]]] = []

    if plan.include_literature:
        literature_jobs.append(
            (SourceType.PUBMED.value, {"query": plan.pubmed_query, "max_records": cap})
        )
        if plan.pmc_query:
            literature_jobs.append(
                (
                    SourceType.PMC.value,
                    {"query": plan.pmc_query, "max_records": min(cap, 40)},
                )
            )
    if plan.include_patents and plan.patent_query:
        patent_jobs.append(
            (
                SourceType.PATENTSVIEW.value,
                {"query_text": plan.patent_query, "max_records": cap},
            )
        )
    if plan.include_trials and (plan.trial_conditions or plan.trial_terms):
        params: dict[str, Any] = {"max_records": cap}
        if plan.trial_conditions:
            params["conditions"] = plan.trial_conditions
        if plan.trial_terms:
            params["query_terms"] = plan.trial_terms
        other_jobs.append((SourceType.CLINICALTRIALS.value, params))
    if plan.include_compounds and plan.gene_targets:
        other_jobs.append(
            (
                SourceType.CHEMBL.value,
                {"targets": plan.gene_targets, "max_records": cap},
            )
        )

    if stats.is_patent_question:
        jobs = patent_jobs + literature_jobs + other_jobs
    else:
        jobs = literature_jobs + patent_jobs + other_jobs

    for source, params in jobs:
        try:
            n = ingest_single(source, max_records=params.pop("max_records", cap), **params)
            stats.ingested += n
            if n:
                stats.sources_run.append(source)
        except Exception as exc:
            logger.warning("[auto-ingest] %s failed: %s", source, exc)
            stats.failed_sources.append(source)

    if stats.ingested:
        stats.indexed_chunks = run_indexing(all_docs=False)
    return stats
