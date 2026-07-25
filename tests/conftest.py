"""Shared pytest fixtures and offline test configuration.

All tests here are **offline**: no network, no Ollama, no Qdrant/OpenSearch,
no Postgres/Redis. Heavy ML dependencies (torch, sentence-transformers,
FlagEmbedding, langgraph) are optional and not required — the code paths under
test degrade gracefully or are exercised with fakes/mocks. Tests that would need
live services are marked ``integration`` and excluded from CI via
``-m "not integration"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biointel.common.schemas import (
    Chunk,
    DocType,
    DocumentIds,
    RetrievedChunk,
    SourceType,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json_fixture(name: str) -> dict:
    return json.loads(load_fixture(name))


@pytest.fixture
def make_chunk():
    """Factory for building Chunk objects in tests."""

    def _make(
        chunk_id: str,
        text: str,
        source: SourceType = SourceType.PUBMED,
        doc_type: DocType = DocType.PAPER,
        **ids,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc:{chunk_id}",
            source=source,
            doc_type=doc_type,
            title=f"Title {chunk_id}",
            text=text,
            ids=DocumentIds(**ids),
            source_url=f"http://example.org/{chunk_id}",
        )

    return _make


@pytest.fixture
def make_retrieved(make_chunk):
    """Factory for RetrievedChunk with optional scores."""

    def _make(chunk_id: str, text: str, *, dense=None, sparse=None, fused=None, **kw):
        return RetrievedChunk(
            chunk=make_chunk(chunk_id, text, **kw),
            dense_score=dense,
            sparse_score=sparse,
            fused_score=fused,
        )

    return _make


class FakeLLM:
    """Deterministic stand-in for the Ollama client used across agent tests."""

    model = "fake-llm"

    def __init__(self, chat_responses=None, json_responses=None):
        self._chat = list(chat_responses or [])
        self._json = dict(json_responses or {})
        self.chat_calls: list = []
        self.json_calls: list = []

    def chat(self, messages, **kw):
        self.chat_calls.append((messages, kw))
        if self._chat:
            return self._chat.pop(0)
        return "Default answer [1]."

    def generate_json(self, messages, schema, **kw):
        self.json_calls.append((schema.__name__, messages))
        if schema.__name__ in self._json:
            return self._json[schema.__name__]
        return schema()  # empty valid instance

    def health(self) -> bool:
        return True


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
