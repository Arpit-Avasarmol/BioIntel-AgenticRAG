"""Pydantic schemas for LLM-constrained planning and per-doc-type extraction.

These models are passed to :meth:`OllamaClient.generate_json` so the local model
emits schema-conforming JSON that we then validate. They serve two purposes:

* **Planning** — decompose a question into retrievable sub-questions.
* **Structured extraction** — turn free-text chunks into typed records
  (trials, compounds, targets, patents, papers) that make the agent's reasoning
  auditable and enable downstream contradiction checks on comparable fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
class SubQuestion(BaseModel):
    """One retrievable sub-question, optionally scoped to a source."""

    question: str = Field(description="A focused, self-contained sub-question.")
    source_type: str | None = Field(
        default=None,
        description=(
            "Optional source filter: one of pubmed, pmc, clinicaltrials, chembl, "
            "opentargets, patentsview, google_patents; null if no preference."
        ),
    )


class QueryPlan(BaseModel):
    """The planner's decomposition of the user question."""

    sub_questions: list[SubQuestion] = Field(default_factory=list)


class DynamicIngestPlan(BaseModel):
    """Sources + queries to bootstrap the corpus for a user question."""

    topic: str = Field(description="Short label for the biomedical topic.")
    pubmed_query: str = Field(description="PubMed esearch query string.")
    pmc_query: str | None = Field(
        default=None, description="PMC query; null to skip PMC during auto-ingest."
    )
    patent_query: str | None = Field(
        default=None, description="PatentsView text query; null to skip patents."
    )
    trial_conditions: list[str] = Field(
        default_factory=list, description="ClinicalTrials.gov condition phrases."
    )
    trial_terms: str | None = Field(
        default=None, description="ClinicalTrials.gov intervention/term query."
    )
    gene_targets: list[str] = Field(
        default_factory=list, description="Gene symbols for ChEMBL target lookup."
    )
    include_literature: bool = True
    include_patents: bool = True
    include_trials: bool = True
    include_compounds: bool = True


# --------------------------------------------------------------------------
# Contradiction detection (LLM output)
# --------------------------------------------------------------------------
class ContradictionItem(BaseModel):
    statement_a: str
    statement_b: str
    source_a: str = Field(description="Source number/label for statement_a, e.g. '3'.")
    source_b: str = Field(description="Source number/label for statement_b, e.g. '5'.")
    explanation: str


class ContradictionReport(BaseModel):
    contradictions: list[ContradictionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Per-doc-type structured extraction
# --------------------------------------------------------------------------
class TrialRecord(BaseModel):
    nct_id: str | None = None
    title: str | None = None
    phase: str | None = None
    status: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    primary_outcome: str | None = None
    enrollment: int | None = None
    sponsor: str | None = None


class CompoundRecord(BaseModel):
    chembl_id: str | None = None
    name: str | None = None
    mechanism_of_action: str | None = None
    targets: list[str] = Field(default_factory=list)
    max_phase: str | None = None
    indications: list[str] = Field(default_factory=list)


class TargetRecord(BaseModel):
    symbol: str | None = None
    ensembl_id: str | None = None
    disease: str | None = None
    efo_id: str | None = None
    association_score: float | None = None
    evidence_types: list[str] = Field(default_factory=list)


class PatentRecord(BaseModel):
    patent_no: str | None = None
    title: str | None = None
    assignee: str | None = None
    filing_year: int | None = None
    key_claim: str | None = None
    targets_or_compounds: list[str] = Field(default_factory=list)


class PaperFinding(BaseModel):
    finding: str | None = None
    entities: list[str] = Field(default_factory=list)
    direction: str | None = Field(
        default=None, description="e.g. 'positive', 'negative', 'no effect' if applicable."
    )
    effect_detail: str | None = None


# Map DocType.value -> extraction schema used by the extract node.
EXTRACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "trial": TrialRecord,
    "compound": CompoundRecord,
    "target": TargetRecord,
    "patent": PatentRecord,
    "paper": PaperFinding,
}
