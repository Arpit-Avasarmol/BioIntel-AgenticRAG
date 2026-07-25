"""Redis client + helpers for caching and rate limiting."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import orjson
import redis

from biointel.common.config import settings


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=False)


# ------------------------------------------------------------------------ cache
def cache_key(namespace: str, *parts: str) -> str:
    raw = "||".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324
    return f"biointel:{namespace}:{digest}"


def cache_get(key: str) -> Any | None:
    raw = get_redis().get(key)
    if raw is None:
        return None
    return orjson.loads(raw)


def cache_set(key: str, value: Any, ttl_seconds: int = 3600) -> None:
    get_redis().set(key, orjson.dumps(value), ex=ttl_seconds)


# ------------------------------------------------------------------- rate limit
def rate_limit_ok(identity: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    """Fixed-window rate limiter.

    Returns ``(allowed, remaining)``. Uses an atomic INCR + EXPIRE on first hit.
    """
    r = get_redis()
    key = f"biointel:ratelimit:{identity}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window_seconds)
    remaining = max(0, limit - int(current))
    return (int(current) <= limit, remaining)
