"""Langfuse LLM/agent tracing helpers.

Provides a small facade so the agent can emit traces/spans/generations without
hard-coupling to Langfuse. When Langfuse is disabled, all calls are no-ops.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from functools import lru_cache
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_langfuse():
    """Return a configured Langfuse client, or None when disabled/misconfigured."""
    if not (settings.langfuse_enabled or settings.obs_enabled):
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse enabled but keys missing; tracing disabled.")
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled -> %s", settings.langfuse_host)
        return client
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("Langfuse init failed (continuing without): %s", exc)
        return None


class _NoOpTrace:
    """Fallback trace object with the subset of the Langfuse API we use."""

    def span(self, **_: Any) -> _NoOpTrace:
        return self

    def generation(self, **_: Any) -> _NoOpTrace:
        return self

    def update(self, **_: Any) -> None:
        return None

    def end(self, **_: Any) -> None:
        return None


@contextmanager
def trace_run(name: str, **metadata: Any):
    """Context manager yielding a trace object (Langfuse or no-op)."""
    client = get_langfuse()
    if client is None:
        yield _NoOpTrace()
        return
    trace = client.trace(name=name, metadata=metadata)
    try:
        yield trace
    finally:
        with suppress(Exception):  # pragma: no cover
            client.flush()
