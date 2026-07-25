"""Dense embedding via sentence-transformers (BAAI/bge-* by default).

The model is loaded lazily and cached process-wide so importing this module is
cheap (no torch import at module load). BGE models benefit from a query
instruction prefix for retrieval, which we apply only to queries (not passages).
"""

from __future__ import annotations

from functools import lru_cache

from biointel.common.config import settings
from biointel.common.devices import resolve_torch_device
from biointel.common.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    device = resolve_torch_device(settings.embedding_device)
    logger.info("Loading embedding model %s on %s", settings.embedding_model, device)
    try:
        return SentenceTransformer(settings.embedding_model, device=device)
    except Exception as exc:
        if device != "cpu":
            logger.warning(
                "Embedding model failed on %s (%s); retrying on cpu.", device, exc
            )
            return SentenceTransformer(settings.embedding_model, device="cpu")
        raise


class Embedder:
    """Encodes passages and queries into dense vectors."""

    def __init__(self) -> None:
        self.dim = settings.embedding_dim
        self.batch_size = settings.embedding_batch_size
        self.query_prefix = settings.embedding_query_prefix

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        model = _load_model()
        text = f"{self.query_prefix} {query}".strip() if self.query_prefix else query
        vec = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vec[0].tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
