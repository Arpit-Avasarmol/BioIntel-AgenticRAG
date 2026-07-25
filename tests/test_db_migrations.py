"""Acceptance test 8: schema + migrations.

Verifies that (a) the ORM models declare the expected tables/columns, and (b) the
Alembic migration applies against a throwaway SQLite database and downgrades
cleanly. Uses SQLite (no Postgres needed) so it runs in CI offline.
"""

from __future__ import annotations

import importlib

from sqlalchemy import create_engine, inspect


def test_orm_tables_and_key_columns():
    from biointel.db import models

    tables = models.Base.metadata.tables
    expected = {"documents", "chunks", "chat_sessions", "chat_messages", "audit_log"}
    assert expected <= set(tables), f"missing tables: {expected - set(tables)}"

    # audit_log must carry the full provenance trail.
    audit_cols = set(tables["audit_log"].columns.keys())
    for col in (
        "trace_id",
        "session_id",
        "question",
        "answer",
        "sub_questions",
        "used_chunk_ids",
        "citations",
        "contradictions",
        "verified",
        "model",
        "embedding_model",
        "reranker_model",
        "prompt_version",
        "latency_ms",
        "warnings",
    ):
        assert col in audit_cols, f"audit_log missing column: {col}"


def test_alembic_upgrade_and_downgrade(tmp_path, monkeypatch):
    """Apply the migration on SQLite, assert tables exist, then downgrade."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", url)

    # Force settings + session modules to pick up the SQLite URL.
    from biointel.common import config as config_mod

    config_mod.get_settings.cache_clear()
    importlib.reload(config_mod)

    from alembic.config import Config

    from alembic import command

    cfg = Config("alembic.ini")

    command.upgrade(cfg, "head")
    insp = inspect(create_engine(url))
    tables = set(insp.get_table_names())
    assert {"documents", "chunks", "chat_sessions", "chat_messages", "audit_log"} <= tables

    command.downgrade(cfg, "base")
    insp2 = inspect(create_engine(url))
    remaining = set(insp2.get_table_names()) - {"alembic_version"}
    assert remaining == set(), f"downgrade left tables behind: {remaining}"
