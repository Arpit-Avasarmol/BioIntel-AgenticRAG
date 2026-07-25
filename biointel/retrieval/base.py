"""Abstract store interfaces for dense vectors and keyword search.

``VectorStore`` is implemented by Qdrant (primary) and Milvus (documented
alternative). ``KeywordStore`` is implemented by OpenSearch (BM25). Keeping these
behind interfaces lets the retrieval layer stay backend-agnostic and makes the
"pluggable vector store" claim real and testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from biointel.common.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Dense vector index over chunk embeddings."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the collection/index if it does not exist."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Insert or update chunk vectors + payload. Returns count upserted."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return top_k nearest chunks (with dense_score set)."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def delete_all(self) -> None:
        """Drop all vectors (used by reindex)."""


class KeywordStore(ABC):
    """Sparse/BM25 keyword index over chunk text."""

    @abstractmethod
    def ensure_index(self) -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int: ...

    @abstractmethod
    def search(
        self, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return top_k BM25 matches (with sparse_score set)."""

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_all(self) -> None: ...


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Flatten a Chunk into a store payload (JSON-serializable)."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source.value,
        "doc_type": chunk.doc_type.value,
        "title": chunk.title,
        "text": chunk.text,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "source_url": chunk.source_url,
        "license": chunk.license,
        "ids": chunk.ids.model_dump(exclude_none=True),
        "date": chunk.date.isoformat() if chunk.date else None,
        "token_count": chunk.token_count,
        "extra": chunk.extra,
    }


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from a stored payload."""
    from biointel.common.schemas import DocType, DocumentIds, SourceType

    date_val = payload.get("date")
    from datetime import date as _date

    parsed_date = None
    if date_val:
        try:
            parsed_date = _date.fromisoformat(date_val)
        except (ValueError, TypeError):
            parsed_date = None

    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        source=SourceType(payload["source"]),
        doc_type=DocType(payload["doc_type"]),
        title=payload.get("title", ""),
        text=payload.get("text", ""),
        section=payload.get("section", ""),
        chunk_index=payload.get("chunk_index", 0),
        source_url=payload.get("source_url", ""),
        license=payload.get("license", ""),
        ids=DocumentIds(**payload.get("ids", {})),
        date=parsed_date,
        token_count=payload.get("token_count", 0),
        extra=payload.get("extra", {}),
    )
