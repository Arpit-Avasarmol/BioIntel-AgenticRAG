"""Local LLM client for Ollama (chat, JSON-structured output, streaming).

All generation goes through a locally hosted Ollama server — no API keys, no data
leaving the host. This module exposes three call styles the agent needs:

* :meth:`OllamaClient.chat` — plain text completion.
* :meth:`OllamaClient.generate_json` — constrained JSON output validated against a
  Pydantic model (used for planning and structured extraction). Ollama's
  ``format="json"`` (or a JSON schema) forces syntactically valid JSON; we then
  validate semantics with Pydantic and retry once on failure.
* :meth:`OllamaClient.stream_chat` — token streaming for the SSE endpoint.

The client is deliberately thin (httpx + tenacity) so it has no LangChain runtime
dependency; LangGraph orchestrates *around* it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from biointel.common.config import settings
from biointel.common.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries or returns unusable output."""


class OllamaClient:
    """Minimal Ollama chat client used by every agent node."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.num_ctx = num_ctx or settings.llm_num_ctx
        self.timeout = timeout or settings.llm_timeout_seconds

    # -- internal ---------------------------------------------------------
    def _options(self, **overrides: Any) -> dict[str, Any]:
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        opts.update(overrides)
        return opts

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    # -- public API -------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        """Return the assistant text for a list of chat ``messages``."""
        opts = self._options(**options)
        if temperature is not None:
            opts["temperature"] = temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        try:
            data = self._post_chat(payload)
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Ollama chat failed: {exc}") from exc
        return data.get("message", {}).get("content", "").strip()

    def complete(self, prompt: str, system: str | None = None, **options: Any) -> str:
        """Convenience one-shot completion from a prompt (+ optional system)."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **options)

    def generate_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        max_retries: int = 1,
        temperature: float | None = None,
    ) -> T:
        """Generate JSON constrained to ``schema`` and validate with Pydantic.

        Uses Ollama's structured-output ``format`` (the JSON schema of the target
        model) so the server emits schema-conforming JSON, then validates. On a
        parse/validation error we append the error text and retry up to
        ``max_retries`` times before raising :class:`LLMError`.
        """
        opts = self._options()
        if temperature is not None:
            opts["temperature"] = temperature

        json_schema = schema.model_json_schema()
        convo = list(messages)
        last_err: Exception | None = None

        for attempt in range(max_retries + 1):
            payload = {
                "model": self.model,
                "messages": convo,
                "stream": False,
                "format": json_schema,
                "options": opts,
            }
            try:
                data = self._post_chat(payload)
            except httpx.HTTPError as exc:  # pragma: no cover - network dependent
                raise LLMError(f"Ollama JSON call failed: {exc}") from exc

            content = data.get("message", {}).get("content", "").strip()
            try:
                return schema.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_err = exc
                logger.warning(
                    "JSON validation failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                convo = list(messages) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That did not match the required schema. "
                            f"Error: {exc}. Return ONLY valid JSON matching the schema."
                        ),
                    },
                ]

        raise LLMError(f"Could not obtain valid JSON after retries: {last_err}")

    def stream_chat(self, messages: list[dict[str, str]], **options: Any) -> Iterator[str]:
        """Yield assistant text chunks as they arrive (for SSE streaming)."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self._options(**options),
        }
        try:
            with (
                httpx.Client(timeout=self.timeout) as client,
                client.stream("POST", url, json=payload) as resp,
            ):
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Ollama stream failed: {exc}") from exc

    def health(self) -> bool:
        """Return True if the Ollama server responds to /api/tags."""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


def get_llm() -> OllamaClient:
    """Construct an Ollama client from settings."""
    return OllamaClient()
