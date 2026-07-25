"""FastAPI dependencies: API-key auth and per-client rate limiting.

Auth is a simple shared API key sent in the ``X-API-Key`` header — appropriate
for a single-host, self-deployed service. Rate limiting uses the Redis
fixed-window limiter so limits hold across multiple API workers/processes.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from biointel.common.config import settings
from biointel.common.logging import get_logger

logger = get_logger(__name__)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate the ``X-API-Key`` header against the configured key."""
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key (X-API-Key header).",
        )
    return x_api_key


async def enforce_rate_limit(request: Request) -> None:
    """Fixed-window rate limit keyed by client IP.

    This dependency is intentionally independent of :func:`require_api_key` so
    that endpoints declare both explicitly and overriding one in tests never
    silently disables the other. Fails open (allows the request) if Redis is
    unreachable, so a cache outage degrades gracefully rather than taking down
    the API.
    """
    client_ip = request.client.host if request.client else "unknown"
    try:
        from biointel.db.cache import rate_limit_ok

        allowed, remaining = rate_limit_ok(
            client_ip,
            settings.rate_limit_requests,
            settings.rate_limit_window_seconds,
        )
    except Exception as exc:  # pragma: no cover - redis dependent
        logger.warning("Rate limiter unavailable (%s); allowing request.", exc)
        return

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.rate_limit_requests} requests / "
                f"{settings.rate_limit_window_seconds}s."
            ),
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )
    request.state.rate_remaining = remaining
