"""Ingestion orchestration: registry, config-driven runs, persistence.

Connectors are registered by source name. The runner fetches Documents, archives
the raw payload to MinIO, and upserts metadata to Postgres. It is idempotent at
the document level (upsert by doc_id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from biointel.common.logging import get_logger
from biointel.common.schemas import Document, SourceType
from biointel.db.repository import upsert_document
from biointel.db.session import session_scope
from biointel.db.storage import ensure_bucket, put_raw
from biointel.ingestion.base import BaseConnector
from biointel.ingestion.chembl import ChEMBLConnector
from biointel.ingestion.clinicaltrials import ClinicalTrialsConnector
from biointel.ingestion.opentargets import OpenTargetsConnector
from biointel.ingestion.patentsview import PatentsViewConnector
from biointel.ingestion.pmc import PMCConnector
from biointel.ingestion.pubmed import PubMedConnector

logger = get_logger(__name__)

# Registry: source name -> factory callable(**params) -> BaseConnector
CONNECTORS: dict[str, type[BaseConnector]] = {
    SourceType.PUBMED.value: PubMedConnector,
    SourceType.PMC.value: PMCConnector,
    SourceType.CLINICALTRIALS.value: ClinicalTrialsConnector,
    SourceType.CHEMBL.value: ChEMBLConnector,
    SourceType.OPENTARGETS.value: OpenTargetsConnector,
    SourceType.PATENTSVIEW.value: PatentsViewConnector,
    # google_patents added lazily (optional dep) in _build_connector.
}


def _build_connector(source: str, params: dict[str, Any]) -> BaseConnector:
    if source == SourceType.GOOGLE_PATENTS.value:
        from biointel.ingestion.google_patents import GooglePatentsConnector

        return GooglePatentsConnector(**params)
    if source not in CONNECTORS:
        raise ValueError(f"Unknown source '{source}'. Known: {sorted(CONNECTORS)}")
    return CONNECTORS[source](**params)


def _persist(doc: Document) -> None:
    """Archive raw + upsert metadata."""
    # Archive the normalized document JSON as the raw reference (auditable).
    key = f"{doc.source.value}/{doc.doc_id.replace(':', '_')}.json"
    try:
        ensure_bucket()
        put_raw(key, doc.model_dump(mode="json"))
        doc.raw_ref = key
    except Exception as exc:  # pragma: no cover - MinIO optional at ingest edge
        logger.warning("MinIO archive failed for %s: %s", doc.doc_id, exc)
    with session_scope() as session:
        upsert_document(session, doc)


def ingest_single(
    source: str, query: str | None = None, max_records: int = 200, **extra: Any
) -> int:
    """Ingest from one source with ad-hoc params."""
    params: dict[str, Any] = {"max_records": max_records, **extra}
    # Map the generic --query onto each connector's expected argument.
    if query:
        if source in (SourceType.PUBMED.value, SourceType.PMC.value):
            params["query"] = query
        elif source == SourceType.CLINICALTRIALS.value:
            params.setdefault("conditions", [query])
        elif source in (SourceType.PATENTSVIEW.value, SourceType.GOOGLE_PATENTS.value):
            params["query_text"] = query
        elif source == SourceType.CHEMBL.value:
            params["targets"] = [t.strip() for t in query.split(",")]
        elif source == SourceType.OPENTARGETS.value:
            params["diseases"] = [d.strip() for d in query.split(",")]

    connector = _build_connector(source, params)
    n = 0
    try:
        for doc in connector.run(max_records=max_records):
            _persist(doc)
            n += 1
    except Exception as exc:
        logger.warning("[ingest] %s failed after %d doc(s): %s", source, n, exc)
        if n == 0:
            raise
    logger.info("[ingest] %s -> %d documents persisted", source, n)
    return n


def ingest_from_config(config_path: str | Path) -> int:
    """Ingest all enabled sources defined in a YAML corpus config."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    sources_cfg: dict[str, dict[str, Any]] = cfg.get("sources", {})
    global_cap = cfg.get("max_records_per_source")

    total = 0
    for source, scfg in sources_cfg.items():
        if not scfg.get("enabled", False):
            logger.info("[ingest] skipping disabled source: %s", source)
            continue
        params = {k: v for k, v in scfg.items() if k not in ("enabled",)}
        max_records = params.pop("max_records", None) or global_cap or 200
        params["max_records"] = max_records
        # Some connectors use differently-named caps.
        if source == SourceType.OPENTARGETS.value:
            params.setdefault("max_targets", params.pop("max_records", max_records))
        try:
            connector = _build_connector(source, params)
        except (RuntimeError, ValueError) as exc:
            logger.warning("[ingest] %s unavailable: %s", source, exc)
            continue
        n = 0
        try:
            for doc in connector.run(max_records=max_records):
                _persist(doc)
                n += 1
        except Exception as exc:
            logger.warning("[ingest] %s failed after %d doc(s): %s", source, n, exc)
            total += n
            continue
        logger.info("[ingest] %s -> %d docs", source, n)
        total += n
    logger.info("[ingest] TOTAL persisted: %d", total)
    return total
