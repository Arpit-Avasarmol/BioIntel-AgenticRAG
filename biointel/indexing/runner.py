"""Indexing orchestration: Document -> chunks -> embeddings -> stores.

Pulls not-yet-indexed documents from Postgres, reconstructs Documents, chunks
them, embeds the chunks, upserts into Qdrant + OpenSearch, records chunk metadata,
and marks documents indexed. Idempotent via deterministic chunk IDs.
"""

from __future__ import annotations

from datetime import date

from biointel.common.logging import get_logger
from biointel.common.schemas import (
    DocType,
    Document,
    DocumentIds,
    Entities,
    Section,
    SourceType,
)
from biointel.db.models import DocumentRecord
from biointel.db.repository import (
    get_all_documents,
    get_unindexed_documents,
    mark_indexed,
    record_chunks,
)
from biointel.db.session import session_scope
from biointel.indexing.chunker import chunk_document
from biointel.indexing.embedder import get_embedder
from biointel.retrieval.factory import get_keyword_store, get_vector_store

logger = get_logger(__name__)


def _record_to_document(rec: DocumentRecord) -> Document:
    """Rebuild a Document from its Postgres row."""
    meta = rec.doc_metadata or {}
    parsed_date = None
    if meta.get("date"):
        try:
            parsed_date = date.fromisoformat(meta["date"])
        except (ValueError, TypeError):
            parsed_date = None
    sections = [Section(**s) for s in meta.get("sections", [])]
    return Document(
        doc_id=rec.doc_id,
        source=SourceType(rec.source),
        doc_type=DocType(rec.doc_type),
        title=rec.title or "",
        abstract=rec.abstract or "",
        sections=sections,
        authors=meta.get("authors", []),
        date=parsed_date,
        ids=DocumentIds(**meta.get("ids", {})),
        entities=Entities(**meta.get("entities", {})),
        source_url=rec.source_url or "",
        license=rec.license or "",
        raw_ref=rec.raw_ref,
        extra=meta.get("extra", {}),
    )


def run_indexing(all_docs: bool = False, source: str | None = None, reindex: bool = False) -> int:
    """Index documents into the vector + keyword stores. Returns chunk count."""
    vstore = get_vector_store()
    kstore = get_keyword_store()
    vstore.ensure_collection()
    kstore.ensure_index()

    if reindex:
        logger.warning("Reindex requested: clearing stores")
        vstore.delete_all()
        kstore.delete_all()

    embedder = get_embedder()
    total_chunks = 0

    with session_scope() as session:
        if reindex or all_docs:
            records = get_all_documents(session, source=source)
        else:
            records = get_unindexed_documents(session, source=source)
        logger.info("Indexing %d documents", len(records))

        for rec in records:
            doc = _record_to_document(rec)
            chunks = chunk_document(doc)
            if not chunks:
                mark_indexed(session, doc.doc_id, 0)
                continue
            vectors = embedder.encode_passages([c.text for c in chunks])
            vstore.upsert(chunks, vectors)
            kstore.upsert(chunks)
            record_chunks(session, chunks)
            mark_indexed(session, doc.doc_id, len(chunks))
            total_chunks += len(chunks)
            logger.info("  indexed %s (%d chunks)", doc.doc_id, len(chunks))

    logger.info("Indexing complete: %d chunks", total_chunks)
    return total_chunks
