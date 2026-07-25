"""FastAPI application: the HTTP surface for BioIntel Agent.

Endpoints
---------
* ``GET  /health``        — liveness + resolved model config + component status.
* ``GET  /metrics``       — Prometheus exposition (scraped by the obs stack).
* ``POST /query``         — run the agent, return a verified, cited answer.
* ``POST /query/stream``  — Server-Sent Events: stream tokens then a final payload.
* ``POST /ingest``        — pull + (optionally) index docs from one connector.
* ``GET  /documents``     — list ingested documents.

Every non-health endpoint requires the ``X-API-Key`` header and is rate limited.
Answers are cached in Redis (keyed by question+filters+prompt_version) and every
answer is written to the audit log with full provenance.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from biointel import __version__
from biointel.api.deps import enforce_rate_limit, require_api_key
from biointel.api.schemas import (
    DocumentsResponse,
    DocumentSummary,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.db.cache import cache_get, cache_key, cache_set
from biointel.db.session import session_scope
from biointel.obs.telemetry import (
    QUERY_COUNTER,
    QUERY_LATENCY,
    init_telemetry,
    instrument_fastapi,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry()
    logger.info(
        "BioIntel API starting (profile=%s, llm=%s, embedding=%s)",
        settings.model_profile,
        settings.llm_model,
        settings.embedding_model,
    )
    yield
    logger.info("BioIntel API shutting down.")


app = FastAPI(
    title="BioIntel Agent API",
    version=__version__,
    description="Agentic RAG for drug discovery & patent intelligence.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.api_cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
instrument_fastapi(app)


# --------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe with resolved config and best-effort component checks."""
    components: dict[str, str] = {}

    # LLM (Ollama) reachability — best effort, never fails the probe.
    try:
        from biointel.agent.llm import get_llm

        components["ollama"] = "up" if get_llm().health() else "down"
    except Exception:  # pragma: no cover
        components["ollama"] = "unknown"

    return HealthResponse(
        status="ok",
        version=__version__,
        model_profile=settings.model_profile,
        llm_model=settings.llm_model or "",
        embedding_model=settings.embedding_model or "",
        components=components,
    )


@app.get("/metrics", tags=["ops"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics exposition."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------- query
def _answer_cache_key(req: QueryRequest) -> str:
    payload = json.dumps(
        {"q": req.question, "f": req.filters or {}, "k": req.top_k},
        sort_keys=True,
    )
    return cache_key("answer", settings.prompt_version, payload)


def _run_and_persist(req: QueryRequest, trace_id: str) -> tuple[QueryResponse, dict]:
    """Run the agent, persist audit + chat, return response and a cacheable dict."""
    from biointel.agent.graph import run_agent
    from biointel.db import repository as repo

    started = time.perf_counter()
    answer = run_agent(
        req.question,
        top_k=req.top_k,
        filters=req.filters,
        trace_id=trace_id,
        auto_ingest=req.auto_ingest,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    session_id = req.session_id
    with session_scope() as db:
        session_id = repo.ensure_session(db, session_id, title=req.question[:60])
        repo.add_message(db, session_id, "user", req.question)
        repo.add_message(
            db,
            session_id,
            "assistant",
            answer.answer,
            citations=[c.model_dump() for c in answer.citations],
        )
        repo.write_audit(
            db,
            answer,
            trace_id=trace_id,
            session_id=session_id,
            embedding_model=settings.embedding_model or "",
            reranker_model=settings.reranker_model or "",
            prompt_version=settings.prompt_version,
            latency_ms=latency_ms,
        )

    resp = QueryResponse(
        answer=answer,
        trace_id=trace_id,
        session_id=session_id,
        latency_ms=latency_ms,
        cached=False,
    )
    return resp, resp.model_dump()


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["agent"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Run the agent and return a verified, cited answer (cached when possible)."""
    from biointel.db import repository as repo

    trace_id = repo.new_trace_id()

    # Serve from cache when enabled and permitted.
    if settings.answer_cache_enabled and req.use_cache:
        try:
            cached = cache_get(_answer_cache_key(req))
        except Exception:  # pragma: no cover - redis dependent
            cached = None
        if cached:
            cached["cached"] = True
            cached["trace_id"] = trace_id
            QUERY_COUNTER.labels(status="cache_hit").inc()
            return QueryResponse.model_validate(cached)

    try:
        with QUERY_LATENCY.time():
            resp, cacheable = _run_and_persist(req, trace_id)
    except Exception as exc:
        QUERY_COUNTER.labels(status="error").inc()
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    QUERY_COUNTER.labels(status="ok").inc()
    if settings.answer_cache_enabled:
        with suppress(Exception):  # pragma: no cover
            cache_set(_answer_cache_key(req), cacheable, settings.answer_cache_ttl_seconds)
    return resp


@app.post(
    "/query/stream",
    tags=["agent"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer via SSE.

    The agent's retrieval/verification is not itself token-streamed, so we run the
    full agent, then stream the final answer text token-by-token followed by a
    terminal ``event: result`` carrying the complete structured payload (citations,
    contradictions, verification). This gives a responsive UI while preserving the
    audited, verified final object.
    """
    from biointel.agent.graph import run_agent
    from biointel.db import repository as repo

    trace_id = repo.new_trace_id()

    async def event_gen() -> AsyncIterator[bytes]:
        started = time.perf_counter()
        try:
            answer = run_agent(
                req.question, top_k=req.top_k, filters=req.filters, trace_id=trace_id,
                auto_ingest=req.auto_ingest,
            )
        except Exception as exc:  # pragma: no cover
            err = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {err}\n\n".encode()
            return
        latency_ms = int((time.perf_counter() - started) * 1000)

        # Stream the answer text in word chunks (SSE ``token`` events).
        for tok in answer.answer.split(" "):
            yield f"event: token\ndata: {json.dumps(tok + ' ')}\n\n".encode()

        session_id = req.session_id
        with session_scope() as db:
            session_id = repo.ensure_session(db, session_id, title=req.question[:60])
            repo.add_message(db, session_id, "user", req.question)
            repo.add_message(
                db,
                session_id,
                "assistant",
                answer.answer,
                citations=[c.model_dump() for c in answer.citations],
            )
            repo.write_audit(
                db,
                answer,
                trace_id=trace_id,
                session_id=session_id,
                embedding_model=settings.embedding_model or "",
                reranker_model=settings.reranker_model or "",
                prompt_version=settings.prompt_version,
                latency_ms=latency_ms,
            )
        QUERY_COUNTER.labels(status="ok").inc()

        payload = QueryResponse(
            answer=answer,
            trace_id=trace_id,
            session_id=session_id,
            latency_ms=latency_ms,
            cached=False,
        ).model_dump()
        yield f"event: result\ndata: {json.dumps(payload)}\n\n".encode()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# --------------------------------------------------------------------- ingest
@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["data"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest from one connector and optionally index the results."""
    from biointel.ingestion.runner import ingest_single

    try:
        n_ingested = ingest_single(req.source, query=req.query, max_records=req.max_records)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=400, detail=f"Ingestion error: {exc}") from exc

    n_indexed = 0
    if req.index and n_ingested:
        try:
            from biointel.indexing.runner import run_indexing

            n_indexed = run_indexing(source=req.source)
        except Exception as exc:
            logger.exception("Indexing failed")
            raise HTTPException(status_code=500, detail=f"Indexing error: {exc}") from exc

    return IngestResponse(
        source=req.source,
        ingested=n_ingested,
        indexed=n_indexed,
        message=f"Ingested {n_ingested} document(s); indexed {n_indexed} chunk-set(s).",
    )


@app.get(
    "/documents",
    response_model=DocumentsResponse,
    tags=["data"],
    dependencies=[Depends(require_api_key)],
)
async def documents(source: str | None = None, limit: int = 100) -> DocumentsResponse:
    """List ingested documents (optionally filtered by source)."""
    from biointel.db import repository as repo

    with session_scope() as db:
        rows = repo.get_all_documents(db, source=source, limit=limit)
        total = repo.count_documents(db)
        docs = [
            DocumentSummary(
                doc_id=r.doc_id,
                source=r.source,
                doc_type=r.doc_type,
                title=r.title or "",
                license=r.license or "",
                indexed=bool(r.indexed),
                n_chunks=r.n_chunks or 0,
            )
            for r in rows
        ]
    return DocumentsResponse(total=total, documents=docs)
