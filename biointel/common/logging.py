"""Structured logging setup using rich for readable local logs."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from biointel.common.config import settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "opensearch", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
