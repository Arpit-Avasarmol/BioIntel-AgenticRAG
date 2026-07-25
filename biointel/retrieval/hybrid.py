"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, then reranked.

Pipeline
--------
1. Run dense (Qdrant) and sparse/BM25 (OpenSearch) search **in parallel**, each
   returning its own ranked list of candidate chunks.
2. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF). RRF is rank-based
   rather than score-based, so it is robust to the fact that cosine similarity and
   BM25 live on completely different scales — no score normalization needed.
3. Rerank the fused candidates with a cross-encoder (see ``reranker.py``) and
   return the top-k for the agent to reason over.

The RRF math is factored into :func:`reciprocal_rank_fusion` as a pure function so
it can be unit-tested offline without any vector store, embedder, or GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import RetrievedChunk

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists of chunks into one, using RRF.

    For each list, a chunk at 1-based rank ``r`` contributes ``weight / (k + r)``
    to its fused score. Scores for the same chunk (matched by ``chunk_id``) are
    summed across lists. The original per-modality scores (dense/sparse) are
    preserved on the surviving ``RetrievedChunk`` so downstream code and the UI
    can show why a chunk was retrieved.

    Parameters
    ----------
    ranked_lists:
        One ranked list per retrieval modality. Order within each list defines the
        rank; list position 0 is rank 1.
    k:
        RRF constant. Larger ``k`` flattens the contribution of top ranks (less
        weight on being #1); the literature default is 60.
    weights:
        Optional per-list weights (same length as ``ranked_lists``). Defaults to
        1.0 for every list.

    Returns
    -------
    A single list sorted by descending fused score, with ``fused_score`` and
    ``rank`` populated on each item.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match number of ranked_lists")

    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for lst, weight in zip(ranked_lists, weights, strict=True):
        for rank, rc in enumerate(lst, start=1):
            cid = rc.chunk.chunk_id
            contribution = weight / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution

            if cid not in fused:
                # First time we see this chunk: keep it as the canonical entry.
                fused[cid] = rc
            else:
                # Merge per-modality scores so a chunk found by both engines
                # carries both its dense and sparse evidence.
                existing = fused[cid]
                if rc.dense_score is not None:
                    existing.dense_score = rc.dense_score
                if rc.sparse_score is not None:
                    existing.sparse_score = rc.sparse_score

    for cid, rc in fused.items():
        rc.fused_score = scores[cid]

    ordered = sorted(fused.values(), key=lambda c: c.fused_score or 0.0, reverse=True)
    for i, rc in enumerate(ordered):
        rc.rank = i + 1
    return ordered


class HybridRetriever:
    """Orchestrates dense + sparse retrieval, RRF fusion, and cross-encoder rerank."""

    def __init__(
        self,
        vector_store: Any = None,
        keyword_store: Any = None,
        embedder: Any = None,
        reranker: Any = None,
    ) -> None:
        # Lazy factory imports so constructing the retriever doesn't force torch
        # or client connections at import time (keeps tests/CI light).
        if vector_store is None or keyword_store is None:
            from biointel.retrieval.factory import get_keyword_store, get_vector_store

            vector_store = vector_store or get_vector_store()
            keyword_store = keyword_store or get_keyword_store()
        if embedder is None:
            from biointel.indexing.embedder import get_embedder

            embedder = get_embedder()
        if reranker is None:
            from biointel.retrieval.reranker import get_reranker

            reranker = get_reranker()

        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Full hybrid retrieval for one query string.

        Parameters
        ----------
        query:
            Natural-language query (a sub-question from the agent, typically).
        top_k:
            Number of chunks to return after reranking. Defaults to
            ``settings.retrieval_final_top_k``.
        filters:
            Metadata filters applied by both stores, e.g.
            ``{"source": "pubmed"}``, ``{"doc_type": ["trial", "paper"]}``, or
            date bounds ``{"date_from": "2020-01-01"}``.
        rerank:
            If ``False``, skip the cross-encoder and return the RRF order (useful
            for ablation / debugging).
        """
        final_k = top_k or settings.retrieval_final_top_k
        d_k = dense_top_k or settings.retrieval_dense_top_k
        s_k = sparse_top_k or settings.retrieval_sparse_top_k

        query_vector = self.embedder.encode_query(query)

        # Run both retrievers concurrently; they hit independent services.
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self.vector_store.search, query_vector, d_k, filters)
            sparse_future = pool.submit(self.keyword_store.search, query, s_k, filters)
            dense_hits = self._safe_result(dense_future, "dense")
            sparse_hits = self._safe_result(sparse_future, "sparse")

        logger.info(
            "Hybrid retrieve: %d dense, %d sparse candidates for query=%r",
            len(dense_hits),
            len(sparse_hits),
            query[:80],
        )

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=settings.rrf_k)
        if not fused:
            return []

        if not rerank:
            return fused[:final_k]

        return self.reranker.rerank(query, fused, top_k=final_k)

    @staticmethod
    def _safe_result(future: Any, label: str) -> list[RetrievedChunk]:
        """Never let one engine failing take down the whole query."""
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("%s retrieval failed: %s", label, exc)
            return []


def get_hybrid_retriever() -> HybridRetriever:
    """Construct a retriever wired to the configured stores/models."""
    return HybridRetriever()
