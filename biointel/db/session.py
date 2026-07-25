"""Database engine + session management (SQLAlchemy 2.0, sync)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from biointel.common.config import settings

_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine():
    """Lazily create the process-wide SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.resolved_database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session (no auto-commit; caller commits)."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
