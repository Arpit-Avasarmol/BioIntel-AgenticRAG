"""Repository helpers: persistence operations used by ingestion, indexing, API.

Keeps ORM details out of the business logic and centralizes upsert semantics.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from biointel.common.schemas import AgentAnswer, Chunk, Document
from biointel.db.models import (
    AuditLog,
    ChatMessage,
    ChatSession,
    ChunkRecord,
    DocumentRecord,
)


# --------------------------------------------------------------------- documents
def upsert_document(session: Session, doc: Document) -> DocumentRecord:
    """Insert or update a document by ``doc_id``."""
    existing = session.execute(
        select(DocumentRecord).where(DocumentRecord.doc_id == doc.doc_id)
    ).scalar_one_or_none()

    metadata = {
        "ids": doc.ids.model_dump(exclude_none=True),
        "entities": doc.entities.model_dump(),
        "authors": doc.authors,
        "date": doc.date.isoformat() if doc.date else None,
        "sections": [s.model_dump() for s in doc.sections],
        "extra": doc.extra,
    }

    if existing:
        existing.title = doc.title
        existing.abstract = doc.abstract
        existing.source_url = doc.source_url
        existing.license = doc.license
        existing.raw_ref = doc.raw_ref
        existing.doc_metadata = metadata
        return existing

    record = DocumentRecord(
        doc_id=doc.doc_id,
        source=doc.source.value,
        doc_type=doc.doc_type.value,
        title=doc.title,
        abstract=doc.abstract,
        source_url=doc.source_url,
        license=doc.license,
        raw_ref=doc.raw_ref,
        doc_metadata=metadata,
        indexed=False,
    )
    session.add(record)
    return record


def get_unindexed_documents(
    session: Session, source: str | None = None, limit: int | None = None
) -> list[DocumentRecord]:
    stmt = select(DocumentRecord).where(DocumentRecord.indexed.is_(False))
    if source:
        stmt = stmt.where(DocumentRecord.source == source)
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars().all())


def get_all_documents(
    session: Session, source: str | None = None, limit: int | None = None
) -> list[DocumentRecord]:
    stmt = select(DocumentRecord)
    if source:
        stmt = stmt.where(DocumentRecord.source == source)
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars().all())


def mark_indexed(session: Session, doc_id: str, n_chunks: int) -> None:
    rec = session.execute(
        select(DocumentRecord).where(DocumentRecord.doc_id == doc_id)
    ).scalar_one_or_none()
    if rec:
        rec.indexed = True
        rec.n_chunks = n_chunks


def record_chunks(session: Session, chunks: list[Chunk]) -> None:
    """Persist chunk metadata (idempotent-ish: skips existing chunk_ids)."""
    if not chunks:
        return
    existing = set(
        session.execute(
            select(ChunkRecord.chunk_id).where(
                ChunkRecord.chunk_id.in_([c.chunk_id for c in chunks])
            )
        ).scalars()
    )
    for c in chunks:
        if c.chunk_id in existing:
            continue
        session.add(
            ChunkRecord(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source=c.source.value,
                section=c.section,
                chunk_index=c.chunk_index,
                token_count=c.token_count,
            )
        )


def count_documents(session: Session) -> int:
    return session.query(DocumentRecord).count()


def count_chunks(session: Session) -> int:
    return session.query(ChunkRecord).count()


# ------------------------------------------------------------------------- audit
def write_audit(
    session: Session,
    answer: AgentAnswer,
    *,
    trace_id: str,
    session_id: str | None,
    embedding_model: str,
    reranker_model: str,
    prompt_version: str,
    latency_ms: int,
) -> AuditLog:
    entry = AuditLog(
        trace_id=trace_id,
        session_id=session_id,
        question=answer.query,
        answer=answer.answer,
        sub_questions=answer.sub_questions,
        used_chunk_ids=answer.used_chunks,
        citations=[c.model_dump() for c in answer.citations],
        contradictions=[c.model_dump() for c in answer.contradictions],
        verified=answer.verified,
        model=answer.model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
        warnings=answer.warnings,
    )
    session.add(entry)
    return entry


# -------------------------------------------------------------------- chat state
def ensure_session(session: Session, session_id: str | None, title: str = "") -> str:
    """Return an existing/created chat session id."""
    sid = session_id or uuid.uuid4().hex
    existing = session.execute(
        select(ChatSession).where(ChatSession.session_id == sid)
    ).scalar_one_or_none()
    if not existing:
        session.add(ChatSession(session_id=sid, title=title or "New chat"))
    return sid


def add_message(
    session: Session, session_id: str, role: str, content: str, citations: list | None = None
) -> None:
    session.add(
        ChatMessage(session_id=session_id, role=role, content=content, citations=citations or [])
    )


def new_trace_id() -> str:
    return f"tr_{int(time.time())}_{uuid.uuid4().hex[:8]}"
