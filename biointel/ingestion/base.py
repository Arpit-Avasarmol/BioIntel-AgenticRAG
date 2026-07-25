"""Base connector interface + shared HTTP client with retry/backoff.

Every source connector subclasses :class:`BaseConnector` and implements
``fetch`` (yield raw records) and ``normalize`` (raw -> :class:`Document`).
This makes new sources pluggable and keeps the ingestion runner source-agnostic.

Design principle: **official APIs / bulk downloads only — never HTML scraping.**
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import Document, SourceType

logger = get_logger(__name__)

# Exceptions worth retrying (transient network / server issues).
_RETRYABLE = (httpx.TransportError,)


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _default_user_agent() -> str:
    email = settings.ncbi_email or "contact@example.com"
    return f"{settings.ncbi_tool}/0.1 (mailto:{email})"


class RateLimiter:
    """Simple minimum-interval throttle (requests per second)."""

    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class HttpClient:
    """Thin wrapper around httpx with polite rate limiting + retry/backoff."""

    def __init__(
        self,
        base_url: str = "",
        rps: float = 3.0,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._limiter = RateLimiter(rps)
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers or {"User-Agent": _default_user_agent()},
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE) | retry_if_exception(_is_retryable_http),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self._limiter.wait()
        resp = self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    @retry(
        retry=retry_if_exception_type(_RETRYABLE) | retry_if_exception(_is_retryable_http),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self._limiter.wait()
        resp = self._client.post(url, **kwargs)
        resp.raise_for_status()
        return resp

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class BaseConnector(ABC):
    """Abstract source connector.

    Subclasses set :attr:`source` and :attr:`doc_type` and implement
    :meth:`fetch` and :meth:`normalize`.
    """

    source: SourceType
    license: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def fetch(self) -> Iterator[dict[str, Any]]:
        """Yield raw source records (dicts). No normalization here."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> Document | None:
        """Convert one raw record into a normalized Document (or None to skip)."""
        raise NotImplementedError

    def run(self, max_records: int | None = None) -> Iterable[Document]:
        """Fetch + normalize, yielding Documents up to ``max_records``."""
        count = 0
        for raw in self.fetch():
            doc = self.normalize(raw)
            if doc is None:
                continue
            yield doc
            count += 1
            if max_records and count >= max_records:
                break
        logger.info("[%s] normalized %d documents", self.source.value, count)
