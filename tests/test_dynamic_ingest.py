"""Tests for query-driven dynamic ingest planning."""

from __future__ import annotations

from biointel.ingestion.dynamic import (
    extract_primary_term,
    heuristic_ingest_plan,
    infer_retrieval_context,
    is_patent_question,
    question_terms,
)
from biointel.ingestion.dynamic import IngestStats


def test_heuristic_plan_detects_patents_and_genes():
    plan = heuristic_ingest_plan("What patents claim KRAS inhibitors?")
    assert "KRAS" in plan.gene_targets
    assert plan.include_patents is True
    assert plan.patent_query == "KRAS"
    assert "KRAS" in plan.pubmed_query
    assert "patent" in plan.pubmed_query.lower()


def test_heuristic_plan_extracts_azithromycin_for_patent_question():
    plan = heuristic_ingest_plan("What patents claim Azithromycin?")
    assert extract_primary_term("What patents claim Azithromycin?") == "Azithromycin"
    assert plan.patent_query == "Azithromycin"
    assert "Azithromycin" in plan.pubmed_query
    assert "patent" in plan.pubmed_query.lower()
    assert is_patent_question("What patents claim Azithromycin?")


def test_question_terms_include_primary_drug():
    terms = question_terms("What patents claim Azithromycin?")
    assert "azithromycin" in terms


def test_infer_retrieval_context_filters_patents_when_ingested():
    stats = IngestStats(
        is_patent_question=True,
        sources_run=["patentsview"],
        topic="Azithromycin",
    )
    ctx = infer_retrieval_context("What patents claim Azithromycin?", stats)
    assert ctx.filters.get("doc_type") == "patent"
    assert "azithromycin" in ctx.required_terms


def test_heuristic_plan_includes_trials_for_clinical_questions():
    plan = heuristic_ingest_plan("Which phase 3 trials study IL-23 in Crohn disease?")
    assert plan.include_trials is True
    assert plan.trial_conditions
    assert plan.trial_terms
