"""Acceptance test 7: API contract with a mocked agent (no live services).

Validates auth, the query/stream/documents/health/metrics endpoints, answer
caching, and request validation using FastAPI's TestClient. The agent, DB, and
cache are mocked so the HTTP contract is tested in isolation.
"""

from __future__ import annotations

import contextlib

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import biointel.agent.graph as graphmod
import biointel.agent.llm as llmmod
import biointel.api.app as appmod
import biointel.api.deps as depsmod
import biointel.db.repository as repo
from biointel.common.config import settings
from biointel.common.schemas import AgentAnswer, Citation, SourceType

KEY = settings.api_key


@pytest.fixture
def client(monkeypatch):
    # Fake Ollama health (no network).
    class FakeLLM:
        model = settings.llm_model

        def health(self):
            return True

    monkeypatch.setattr(llmmod, "get_llm", lambda: FakeLLM())

    # Fake agent answer.
    answer = AgentAnswer(
        query="q",
        answer="Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1].",
        citations=[
            Citation(
                marker="[1]",
                chunk_id="c1",
                doc_id="doc:c1",
                source=SourceType.PUBMED,
                source_url="http://x/c1",
                label="pubmed pmid=111",
                quote="q",
            )
        ],
        contradictions=[],
        used_chunks=["c1"],
        sub_questions=["q1", "q2"],
        model=settings.llm_model,
        verified=True,
        warnings=[],
    )
    monkeypatch.setattr(graphmod, "run_agent", lambda *a, **k: answer)

    # Neutralize persistence + cache.
    monkeypatch.setattr(repo, "new_trace_id", lambda: "tr_test_123")
    monkeypatch.setattr(repo, "ensure_session", lambda db, sid, title="": sid or "sess_test")
    monkeypatch.setattr(repo, "add_message", lambda *a, **k: None)
    monkeypatch.setattr(repo, "write_audit", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "session_scope", lambda: contextlib.nullcontext(object()))
    monkeypatch.setattr(appmod, "cache_get", lambda k: None)
    monkeypatch.setattr(appmod, "cache_set", lambda *a, **k: None)

    # Bypass Redis rate limiter (properly annotated so FastAPI treats it as Request).
    async def _no_rl(request: Request, _: str = "x"):
        return None

    appmod.app.dependency_overrides[depsmod.enforce_rate_limit] = _no_rl
    yield TestClient(appmod.app)
    appmod.app.dependency_overrides.clear()


def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"]["ollama"] == "up"


def test_query_requires_api_key(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 401
    assert (
        client.post("/query", json={"question": "hi"}, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )


def test_query_returns_verified_cited_answer(client):
    r = client.post(
        "/query",
        json={"question": "Which IL-23 inhibitors work in Crohn?"},
        headers={"X-API-Key": KEY},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["verified"] is True
    assert body["answer"]["citations"][0]["marker"] == "[1]"
    assert body["trace_id"] == "tr_test_123"
    assert body["cached"] is False
    assert "latency_ms" in body


def test_query_cache_hit(client, monkeypatch):
    r1 = client.post("/query", json={"question": "cached q"}, headers={"X-API-Key": KEY})
    payload = r1.json()
    monkeypatch.setattr(appmod, "cache_get", lambda k: dict(payload))
    r2 = client.post("/query", json={"question": "cached q"}, headers={"X-API-Key": KEY})
    assert r2.status_code == 200 and r2.json()["cached"] is True


def test_query_stream_sse(client):
    r = client.post(
        "/query/stream", json={"question": "stream this please"}, headers={"X-API-Key": KEY}
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "event: token" in r.text and "event: result" in r.text


def test_documents_listing(client, monkeypatch):
    class Row:
        doc_id = "pubmed:111"
        source = "pubmed"
        doc_type = "paper"
        title = "T"
        license = "public-domain"
        indexed = True
        n_chunks = 3

    monkeypatch.setattr(repo, "get_all_documents", lambda db, source=None, limit=100: [Row()])
    monkeypatch.setattr(repo, "count_documents", lambda db: 1)
    r = client.get("/documents", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert r.json()["total"] == 1
    assert r.json()["documents"][0]["doc_id"] == "pubmed:111"


def test_metrics_exposition(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "biointel_queries_total" in r.text


def test_empty_question_validation(client):
    r = client.post("/query", json={"question": ""}, headers={"X-API-Key": KEY})
    assert r.status_code == 422
