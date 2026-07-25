"""OpenSearch BM25 keyword store (sparse side of hybrid retrieval)."""

from __future__ import annotations

from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import Chunk, RetrievedChunk
from biointel.retrieval.base import KeywordStore, chunk_to_payload, payload_to_chunk

logger = get_logger(__name__)

_INDEX_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {"analyzer": {"default": {"type": "english"}}},
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "title": {"type": "text"},
            "text": {"type": "text"},
            "section": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "source_url": {"type": "keyword"},
            "license": {"type": "keyword"},
            "date": {"type": "date", "ignore_malformed": True},
            "token_count": {"type": "integer"},
        }
    },
}


class OpenSearchStore(KeywordStore):
    def __init__(self) -> None:
        from opensearchpy import OpenSearch

        self.index = settings.opensearch_index
        self.client = OpenSearch(
            hosts=[settings.opensearch_url],
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=settings.opensearch_url.startswith("https"),
            verify_certs=settings.opensearch_verify_certs,
            ssl_show_warn=False,
            timeout=60,
        )

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(index=self.index, body=_INDEX_BODY)
        logger.info("Created OpenSearch index %s", self.index)

    def upsert(self, chunks: list[Chunk]) -> int:
        from opensearchpy.helpers import bulk

        if not chunks:
            return 0
        actions = []
        for c in chunks:
            payload = chunk_to_payload(c)
            # Store nested ids/extra as flattened JSON-friendly fields.
            payload.pop("extra", None)
            _id_vals = c.ids.model_dump(exclude_none=True).values()
            payload["ids_flat"] = " ".join(str(v) for v in _id_vals)
            actions.append({"_index": self.index, "_id": c.chunk_id, "_source": payload})
        success, _ = bulk(self.client, actions, refresh=True)
        return success

    def search(
        self, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        must: list[dict[str, Any]] = [
            {"multi_match": {"query": query, "fields": ["title^2", "text", "ids_flat"]}}
        ]
        filt: list[dict[str, Any]] = []
        if filters:
            for key, value in filters.items():
                if isinstance(value, (list, tuple, set)):
                    filt.append({"terms": {key: list(value)}})
                else:
                    filt.append({"term": {key: value}})
        body = {"size": top_k, "query": {"bool": {"must": must, "filter": filt}}}
        resp = self.client.search(index=self.index, body=body)
        results = []
        for rank, hit in enumerate(resp["hits"]["hits"]):
            chunk = payload_to_chunk(hit["_source"])
            results.append(
                RetrievedChunk(chunk=chunk, sparse_score=float(hit["_score"]), rank=rank)
            )
        return results

    def count(self) -> int:
        if not self.client.indices.exists(index=self.index):
            return 0
        return int(self.client.count(index=self.index)["count"])

    def delete_all(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        self.ensure_index()
