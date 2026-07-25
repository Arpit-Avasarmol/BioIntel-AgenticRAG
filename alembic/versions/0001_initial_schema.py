"""initial schema: documents, chunks, chat, audit_log

Revision ID: 0001_initial
Revises:
Create Date: 2025-01-01 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(length=256), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("doc_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("license", sa.String(length=64), nullable=True),
        sa.Column("raw_ref", sa.String(length=512), nullable=True),
        sa.Column("doc_metadata", sa.JSON(), nullable=True),
        sa.Column("indexed", sa.Boolean(), nullable=True),
        sa.Column("n_chunks", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id"),
    )
    op.create_index("ix_documents_doc_id", "documents", ["doc_id"])
    op.create_index("ix_documents_source", "documents", ["source"])
    op.create_index("ix_documents_doc_type", "documents", ["doc_type"])
    op.create_index("ix_documents_indexed", "documents", ["indexed"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=256), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("section", sa.String(length=128), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", name="uq_chunk_id"),
    )
    op.create_index("ix_chunks_chunk_id", "chunks", ["chunk_id"])
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_source", "chunks", ["source"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
    )
    op.create_index("ix_chat_sessions_session_id", "chat_sessions", ["session_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_sessions.session_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sub_questions", sa.JSON(), nullable=True),
        sa.Column("used_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("contradictions", sa.JSON(), nullable=True),
        sa.Column("verified", sa.Boolean(), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("reranker_model", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_trace_id", "audit_log", ["trace_id"])
    op.create_index("ix_audit_log_session_id", "audit_log", ["session_id"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("chunks")
    op.drop_table("documents")
