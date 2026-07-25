"""OpenTelemetry + Prometheus wiring.

All exporters are opt-in (``OTEL_ENABLED`` / ``OBS_ENABLED``). When disabled, the
tracer/meter are cheap no-ops so the base stack runs without Jaeger/Prometheus.
"""

from __future__ import annotations

from functools import lru_cache

from prometheus_client import Counter, Histogram

from biointel.common.config import settings
from biointel.common.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------- Prometheus metrics
QUERY_COUNTER = Counter("biointel_queries_total", "Total agent queries", ["status"])
QUERY_LATENCY = Histogram(
    "biointel_query_latency_seconds",
    "End-to-end agent query latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
RETRIEVAL_LATENCY = Histogram("biointel_retrieval_latency_seconds", "Hybrid retrieval latency")
INGEST_COUNTER = Counter("biointel_documents_ingested_total", "Documents ingested", ["source"])


@lru_cache
def _tracer_provider():
    """Configure and return an OTel tracer provider (once)."""
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled -> %s", settings.otel_exporter_otlp_endpoint)
    return provider


def init_telemetry() -> None:
    """Initialize tracing if enabled. Safe to call multiple times."""
    if settings.otel_enabled or settings.obs_enabled:
        try:
            _tracer_provider()
        except Exception as exc:  # pragma: no cover - optional dep/endpoint
            logger.warning("OTel init failed (continuing without tracing): %s", exc)


def instrument_fastapi(app) -> None:
    """Attach FastAPI OTel instrumentation if enabled."""
    if not (settings.otel_enabled or settings.obs_enabled):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # pragma: no cover
        logger.warning("FastAPI OTel instrumentation failed: %s", exc)


def get_tracer(name: str = "biointel"):
    """Return a tracer (real if enabled, no-op otherwise)."""
    from opentelemetry import trace

    return trace.get_tracer(name)
