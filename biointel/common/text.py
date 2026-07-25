"""Token counting and text helpers shared by chunking + agent prompt budgeting."""

from __future__ import annotations

import functools
import re

import tiktoken

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTINL_RE = re.compile(r"\n{3,}")


@functools.lru_cache(maxsize=4)
def _encoder(name: str = "cl100k_base"):
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    """Approximate token count (cl100k_base — good enough for budgeting)."""
    if not text:
        return 0
    return len(_encoder().encode(text))


def clean_text(text: str) -> str:
    """Collapse redundant whitespace while preserving paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = _WS_RE.sub(" ", text)
    text = _MULTINL_RE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no heavy NLP dep).

    Used by citation verification to check each answer sentence against sources.
    """
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/quote/digit.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most ``max_tokens`` tokens."""
    enc = _encoder()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])
