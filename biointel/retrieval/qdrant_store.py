"""Qdrant dense vector store (primary backend)."""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import Chunk, RetrievedChunk
from biointel.retrieval.base import VectorStore, chunk_to_payload, payload_to_chunk

logger = get_logger(__name__)


def _point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or unsigned int; derive a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore(VectorStore):
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        self.client = QdrantClient(url=settings.qdrant_url, timeout=60)

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        # Payload indexes for fast metadata filtering.
        for field in ("source", "doc_type"):
            # Index may already exist; that is fine.
            with suppress(Exception):  # pragma: no cover
                self.client.create_payload_index(self.collection, field, "keyword")
        logger.info("Created Qdrant collection %s (dim=%d)", self.collection, self.dim)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0
        points = [
            PointStruct(id=_point_id(c.chunk_id), vector=v, payload=chunk_to_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def _build_filter(self, filters: dict[str, Any] | None):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must = []
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                must.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)

    def search(
        self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        results = []
        for rank, h in enumerate(response.points):
            chunk = payload_to_chunk(h.payload)
            results.append(RetrievedChunk(chunk=chunk, dense_score=float(h.score), rank=rank))
        return results

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def delete_all(self) -> None:
        with suppress(Exception):  # pragma: no cover
            self.client.delete_collection(self.collection)
        self.ensure_collection()
