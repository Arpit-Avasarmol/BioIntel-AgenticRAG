"""Core normalized data models shared across ingestion, indexing, retrieval, agent.

Every source connector maps its raw payload to :class:`Document`. Indexing splits
a ``Document`` into :class:`Chunk` objects that carry full provenance so the agent
can verify every citation back to an exact source location.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from datetime import date as date_type
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Canonical source identifiers (one per connector)."""

    PUBMED = "pubmed"
    PMC = "pmc"
    CLINICALTRIALS = "clinicaltrials"
    CHEMBL = "chembl"
    OPENTARGETS = "opentargets"
    PATENTSVIEW = "patentsview"
    GOOGLE_PATENTS = "google_patents"


class DocType(StrEnum):
    """Document categories used for filtering and structured extraction routing."""

    PAPER = "paper"
    TRIAL = "trial"
    COMPOUND = "compound"
    TARGET = "target"
    PATENT = "patent"


class Section(BaseModel):
    """A titled block of text within a document (e.g. Abstract, Methods, Claims)."""

    name: str
    text: str
    order: int = 0


class DocumentIds(BaseModel):
    """External identifiers; all optional. Enables cross-source linking/citation."""

    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    nct_id: str | None = None
    chembl_id: str | None = None
    ensembl_id: str | None = None
    efo_id: str | None = None
    patent_no: str | None = None


class Entities(BaseModel):
    """Lightweight entity annotations captured at ingestion when available."""

    targets: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """Normalized document — the common contract produced by every connector."""

    doc_id: str  # stable, source-prefixed, e.g. "pubmed:12345678"
    source: SourceType
    doc_type: DocType
    title: str = ""
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    date: date_type | None = None
    ids: DocumentIds = Field(default_factory=DocumentIds)
    entities: Entities = Field(default_factory=Entities)
    source_url: str = ""
    license: str = ""  # e.g. "CC-BY-SA-3.0", "CC0-1.0", "public-domain"
    raw_ref: str | None = None  # MinIO object key for the raw payload
    extra: dict[str, Any] = Field(default_factory=dict)  # source-specific metadata
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def full_text(self) -> str:
        """Concatenate title + abstract + ordered sections into one string."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        for sec in sorted(self.sections, key=lambda s: s.order):
            if sec.text.strip():
                header = f"## {sec.name}\n" if sec.name else ""
                parts.append(f"{header}{sec.text}")
        return "\n\n".join(parts).strip()


class Chunk(BaseModel):
    """A retrievable unit derived from a Document, with full provenance."""

    chunk_id: str  # deterministic hash of (doc_id, section, index)
    doc_id: str
    source: SourceType
    doc_type: DocType
    title: str = ""
    text: str
    section: str = ""
    chunk_index: int = 0
    source_url: str = ""
    license: str = ""
    ids: DocumentIds = Field(default_factory=DocumentIds)
    date: date_type | None = None
    token_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, section: str, index: int) -> str:
        raw = f"{doc_id}||{section}||{index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324 (non-crypto)

    def citation_label(self) -> str:
        """Short human-readable provenance label used in answers."""
        bits = [self.source.value]
        for key in ("pmid", "pmcid", "nct_id", "chembl_id", "patent_no", "doi"):
            val = getattr(self.ids, key)
            if val:
                bits.append(f"{key}={val}")
                break
        if self.section:
            bits.append(self.section)
        return " · ".join(bits)


# ---------------------------------------------------------------------------
# Retrieval + agent I/O models
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval with scoring metadata."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0


class Citation(BaseModel):
    """A verified link between an answer sentence and a supporting chunk."""

    marker: str  # e.g. "[1]"
    chunk_id: str
    doc_id: str
    source: SourceType
    source_url: str = ""
    label: str = ""
    quote: str = ""  # supporting snippet from the chunk


class Contradiction(BaseModel):
    """A detected conflict between claims drawn from different sources."""

    statement_a: str
    statement_b: str
    source_a: str
    source_b: str
    explanation: str


class AgentAnswer(BaseModel):
    """Final agent output returned by the API."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    used_chunks: list[str] = Field(default_factory=list)  # chunk_ids consulted
    sub_questions: list[str] = Field(default_factory=list)
    model: str = ""
    verified: bool = False  # did citation verification pass?
    warnings: list[str] = Field(default_factory=list)
