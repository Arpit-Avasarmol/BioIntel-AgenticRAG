"""Cross-encoder reranking via sentence-transformers.

A bi-encoder (the embedder) is fast but scores query and passage independently.
A cross-encoder jointly encodes the (query, passage) pair and is far more
accurate at judging relevance, so we use it to re-order the fused candidate set
before it reaches the LLM. The model is loaded lazily and cached process-wide.

The reranker is optional: if the model cannot be loaded or scored (dependency
absent, device error, version mismatch) the reranker degrades gracefully and
preserves the input order, setting ``rerank_score`` to the incoming fused score.
"""

from __future__ import annotations

from functools import lru_cache

from biointel.common.config import settings
from biointel.common.devices import resolve_torch_device
from biointel.common.logging import get_logger
from biointel.common.schemas import RetrievedChunk

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_reranker():
    """Load and cache the cross-encoder. Returns ``None`` if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover - depends on optional heavy dep
        logger.warning(
            "sentence-transformers CrossEncoder not available (%s); reranking disabled.",
            exc,
        )
        return None

    device = resolve_torch_device(settings.reranker_device)
    logger.info(
        "Loading reranker %s on %s",
        settings.reranker_model,
        device,
    )
    try:
        return CrossEncoder(settings.reranker_model, device=device)
    except Exception as exc:
        if device != "cpu":
            logger.warning("Reranker failed on %s (%s); retrying on cpu.", device, exc)
            try:
                return CrossEncoder(settings.reranker_model, device="cpu")
            except Exception as cpu_exc:  # pragma: no cover
                logger.warning("Failed to load reranker on cpu (%s); disabled.", cpu_exc)
                return None
        logger.warning("Failed to load reranker (%s); reranking disabled.", exc)
        return None


class Reranker:
    """Re-orders candidate chunks by cross-encoder relevance to the query."""

    def __init__(self) -> None:
        self.top_k = settings.reranker_top_k
        self.max_length = settings.reranker_max_length

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return candidates sorted by rerank score, truncated to ``top_k``.

        When the model is unavailable, preserve the incoming order (which is the
        RRF-fused order) and copy ``fused_score`` into ``rerank_score`` so
        downstream code always has a populated score to sort/threshold on.
        """
        limit = top_k or self.top_k
        if not candidates:
            return []

        model = _load_reranker()
        if model is None:
            return self._fallback_order(candidates, limit)

        pairs = [[query, rc.chunk.text] for rc in candidates]
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
            )
            scores = [float(s) for s in raw_scores]
        except Exception as exc:  # pragma: no cover - version / device mismatches
            logger.warning("Reranker scoring failed (%s); using fused order.", exc)
            return self._fallback_order(candidates, limit)

        for rc, score in zip(candidates, scores, strict=False):
            rc.rerank_score = score

        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return self._finalize(ranked[:limit])

    @staticmethod
    def _fallback_order(candidates: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        for rc in candidates:
            if rc.rerank_score is None:
                rc.rerank_score = rc.fused_score if rc.fused_score is not None else 0.0
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return Reranker._finalize(ranked[:limit])

    @staticmethod
    def _finalize(ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
        for i, rc in enumerate(ranked):
            rc.rank = i + 1
        return ranked


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
