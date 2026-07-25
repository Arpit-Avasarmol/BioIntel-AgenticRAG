#!/usr/bin/env python
"""Seed a tiny, fully OFFLINE demo corpus so the whole stack can be exercised
end-to-end without touching any external API.

What it does
------------
1. Reads the six bundled fixtures in ``tests/fixtures/`` (one representative
   record per source: PubMed, PMC, ClinicalTrials, ChEMBL, Open Targets,
   PatentsView).
2. Runs each through its connector's ``normalize()`` to build a canonical
   :class:`~biointel.common.schemas.Document` (identical code path to a live
   ingest — only the network fetch is replaced by the fixture bytes).
3. Persists each Document to Postgres (+ archives the JSON to MinIO if reachable).
4. Chunks, embeds, and upserts everything into Qdrant + OpenSearch.

Prerequisites
-------------
The data services must be running (``make up``) and migrations applied
(``make migrate``). This script needs the embedding model, so the full ML stack
must be installed (``pip install -e .`` on the workstation). It does **not** need
Ollama or GPU — indexing only uses the embedder.

After it finishes you can immediately::

    biointel query "What IL-23 inhibitors are used in inflammatory bowel disease?"

Usage
-----
    python scripts/seed_demo.py            # normal run
    python scripts/seed_demo.py --reindex  # wipe stores first, then re-seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when run as a bare script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biointel.common.logging import get_logger, setup_logging  # noqa: E402
from biointel.common.schemas import Document  # noqa: E402

logger = get_logger("seed_demo")

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _read_json(name: str) -> dict:
    import json

    return json.loads(_read(name))


def build_documents() -> list[Document]:
    """Normalize every fixture into a Document via its real connector."""
    from biointel.ingestion.chembl import ChEMBLConnector
    from biointel.ingestion.clinicaltrials import ClinicalTrialsConnector
    from biointel.ingestion.opentargets import OpenTargetsConnector
    from biointel.ingestion.patentsview import PatentsViewConnector
    from biointel.ingestion.pmc import PMCConnector
    from biointel.ingestion.pubmed import PubMedConnector

    # (connector instance, raw payload) — the raw shape matches what each
    # connector's fetch() yields to normalize().
    jobs = [
        (PubMedConnector(query="seed"), {"xml": _read("pubmed_article.xml")}),
        (PMCConnector(query="seed"), {"xml": _read("pmc_article.xml")}),
        (
            ClinicalTrialsConnector(conditions=["Crohn Disease"]),
            _read_json("clinicaltrials_study.json"),
        ),
        (ChEMBLConnector(targets=["IL23A"]), _read_json("chembl_compound.json")),
        (
            OpenTargetsConnector(diseases=["EFO_0003767"]),
            _read_json("opentargets_assoc.json"),
        ),
        (
            PatentsViewConnector(query_text="IL-23 antibody"),
            _read_json("patentsview_patent.json"),
        ),
    ]

    docs: list[Document] = []
    for connector, raw in jobs:
        try:
            doc = connector.normalize(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("normalize failed for %s: %s", type(connector).__name__, exc)
            continue
        if doc is None:
            logger.warning("%s returned no document (license/OA filter?)", type(connector).__name__)
            continue
        docs.append(doc)
        logger.info("normalized %s  [%s]", doc.doc_id, doc.doc_type.value)
    return docs


def persist(docs: list[Document]) -> None:
    """Archive raw + upsert metadata using the same path as live ingestion."""
    from biointel.ingestion.runner import _persist

    for doc in docs:
        _persist(doc)
    logger.info("persisted %d documents to Postgres/MinIO", len(docs))


def index(reindex: bool) -> int:
    """Chunk + embed + upsert into the vector and keyword stores."""
    from biointel.indexing.runner import run_indexing

    return run_indexing(all_docs=True, reindex=reindex)


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed an offline demo corpus.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear the vector + keyword stores before indexing.",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Building demo documents from offline fixtures in %s", FIXTURES)

    docs = build_documents()
    if not docs:
        logger.error("No documents were built — check the fixtures directory.")
        return 1

    persist(docs)
    n_chunks = index(reindex=args.reindex)

    logger.info("=" * 60)
    logger.info("Seed complete: %d documents, %d chunks indexed.", len(docs), n_chunks)
    logger.info('Try:  biointel query "What IL-23 inhibitors are used in IBD?"')
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
