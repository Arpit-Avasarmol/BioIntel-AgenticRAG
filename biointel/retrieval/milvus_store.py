"""Milvus dense vector store (documented alternative to Qdrant).

Enable by setting ``VECTOR_BACKEND=milvus`` and starting the milvus compose
profile (``docker compose --profile milvus up -d``). Install extra: ``.[milvus]``.
Implements the same :class:`VectorStore` interface as Qdrant so the retrieval
layer is unchanged.
"""

from __future__ import annotations

from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import Chunk, RetrievedChunk
from biointel.retrieval.base import VectorStore, payload_to_chunk

logger = get_logger(__name__)

# Scalar payload fields stored alongside the vector.
_SCALAR_FIELDS = [
    "chunk_id",
    "doc_id",
    "source",
    "doc_type",
    "title",
    "text",
    "section",
    "chunk_index",
    "source_url",
    "license",
    "date",
    "token_count",
]


class MilvusStore(VectorStore):
    def __init__(self) -> None:
        from pymilvus import MilvusClient

        self.collection = settings.milvus_collection
        self.dim = settings.embedding_dim
        self.client = MilvusClient(uri=settings.milvus_uri)

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        # Auto-id disabled; we use the deterministic chunk_id (string PK).
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dim,
            id_type="string",
            primary_field_name="chunk_id",
            vector_field_name="vector",
            metric_type="COSINE",
            max_length=64,
            auto_id=False,
            enable_dynamic_field=True,
        )
        logger.info("Created Milvus collection %s (dim=%d)", self.collection, self.dim)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
        rows = []
        for c, v in zip(chunks, vectors, strict=True):
            row: dict[str, Any] = {
                "chunk_id": c.chunk_id,
                "vector": v,
                "doc_id": c.doc_id,
                "source": c.source.value,
                "doc_type": c.doc_type.value,
                "title": c.title,
                "text": c.text,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "source_url": c.source_url,
                "license": c.license,
                "date": c.date.isoformat() if c.date else "",
                "token_count": c.token_count,
                "ids": c.ids.model_dump(exclude_none=True),
                "extra": c.extra,
            }
            rows.append(row)
        self.client.upsert(collection_name=self.collection, data=rows)
        return len(rows)

    def search(
        self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        expr = None
        if filters:
            clauses = []
            for key, value in filters.items():
                if isinstance(value, (list, tuple, set)):
                    vals = ", ".join(f'"{v}"' for v in value)
                    clauses.append(f"{key} in [{vals}]")
                else:
                    clauses.append(f'{key} == "{value}"')
            expr = " and ".join(clauses)

        res = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            filter=expr or "",
            output_fields=[*_SCALAR_FIELDS, "ids", "extra"],
        )
        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(res[0]):
            entity = hit.get("entity", hit)
            chunk = payload_to_chunk(entity)
            results.append(
                RetrievedChunk(chunk=chunk, dense_score=float(hit.get("distance", 0.0)), rank=rank)
            )
        return results

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def delete_all(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.ensure_collection()
