"""Torch device resolution with safe CPU fallback."""

from __future__ import annotations

from biointel.common.logging import get_logger

logger = get_logger(__name__)


def resolve_torch_device(requested: str) -> str:
    """Return a usable torch device, falling back to CPU when GPU is unavailable."""
    device = (requested or "cpu").lower()
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception as exc:  # pragma: no cover - import / driver errors
            logger.warning("CUDA requested but unavailable (%s); using cpu.", exc)
        return "cpu"
    if device == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover
            pass
        return "cpu"
    return device
