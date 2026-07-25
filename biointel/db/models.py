"""SQLAlchemy ORM models: documents, chunk metadata, chat sessions, audit log.

These tables hold *metadata and provenance*. Raw payloads live in MinIO; vectors
live in Qdrant; searchable text lives in OpenSearch. The ``audit_log`` table makes
every answer reproducible: it stores the question, the exact chunks consulted, the
generated answer, its citations, and the model/prompt versions used.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """One row per normalized source document (metadata + provenance)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    doc_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    source_url: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(64), default="")
    raw_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)  # ids, entities, dates, extra
    indexed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRecord(Base):
    """Metadata for each retrievable chunk (the vector/text live elsewhere)."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("chunk_id", name="uq_chunk_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), index=True)
    doc_id: Mapped[str] = mapped_column(
        String(256), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(32), index=True)
    section: Mapped[str] = mapped_column(String(128), default="")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


class ChatSession(Base):
    """A conversation thread."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(256), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """A single user or assistant turn within a session."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class AuditLog(Base):
    """Audit-ready record of every answered query (full reproducibility trail)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sub_questions: Mapped[list] = mapped_column(JSON, default=list)
    used_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    contradictions: Mapped[list] = mapped_column(JSON, default=list)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str] = mapped_column(String(128), default="")
    embedding_model: Mapped[str] = mapped_column(String(128), default="")
    reranker_model: Mapped[str] = mapped_column(String(128), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
