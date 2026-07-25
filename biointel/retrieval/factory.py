"""Factory functions that instantiate the configured stores.

Centralizes backend selection (Qdrant vs Milvus) so callers never import a
concrete store directly. Instances are cached per process.
"""

from __future__ import annotations

from functools import lru_cache

from biointel.common.config import settings
from biointel.retrieval.base import KeywordStore, VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    if settings.vector_backend == "milvus":
        from biointel.retrieval.milvus_store import MilvusStore

        return MilvusStore()
    from biointel.retrieval.qdrant_store import QdrantStore

    return QdrantStore()


@lru_cache(maxsize=1)
def get_keyword_store() -> KeywordStore:
    from biointel.retrieval.opensearch_store import OpenSearchStore

    return OpenSearchStore()
