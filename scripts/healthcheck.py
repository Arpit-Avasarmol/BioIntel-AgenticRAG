#!/usr/bin/env python
"""Verify that every backing service BioIntel depends on is reachable.

Probes Postgres, Redis, Qdrant, OpenSearch, MinIO and (best-effort) Ollama using
the endpoints from the active configuration. Prints a compact status table and
exits non-zero if any *required* service is unreachable, so it can gate CI or a
``make up`` smoke test.

Ollama is treated as optional here: indexing/ingestion do not need it, only the
agent query path does. Its status is reported but does not fail the check.

Usage
-----
    python scripts/healthcheck.py
    python scripts/healthcheck.py --require-ollama   # also fail if Ollama is down
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biointel.common.config import settings  # noqa: E402

TIMEOUT = 5.0


def check_postgres() -> tuple[bool, str]:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.resolved_database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True, settings.resolved_database_url.split("@")[-1]
    except Exception as exc:
        return False, str(exc)[:80]


def check_redis() -> tuple[bool, str]:
    try:
        import redis

        client = redis.from_url(settings.redis_url, socket_connect_timeout=TIMEOUT)
        client.ping()
        return True, settings.redis_url
    except Exception as exc:
        return False, str(exc)[:80]


def check_qdrant() -> tuple[bool, str]:
    try:
        import httpx

        r = httpx.get(f"{settings.qdrant_url}/readyz", timeout=TIMEOUT)
        if r.status_code < 400:
            return True, settings.qdrant_url
        # older builds expose /healthz or root
        r = httpx.get(settings.qdrant_url, timeout=TIMEOUT)
        return r.status_code < 500, f"{settings.qdrant_url} ({r.status_code})"
    except Exception as exc:
        return False, str(exc)[:80]


def check_opensearch() -> tuple[bool, str]:
    try:
        import httpx

        r = httpx.get(
            f"{settings.opensearch_url}/_cluster/health",
            auth=(settings.opensearch_user, settings.opensearch_password),
            verify=settings.opensearch_verify_certs,
            timeout=TIMEOUT,
        )
        ok = r.status_code < 400
        status = r.json().get("status", "?") if ok else str(r.status_code)
        return ok, f"{settings.opensearch_url} (cluster={status})"
    except Exception as exc:
        return False, str(exc)[:80]


def check_minio() -> tuple[bool, str]:
    try:
        import httpx

        scheme = "https" if settings.minio_secure else "http"
        r = httpx.get(
            f"{scheme}://{settings.minio_endpoint}/minio/health/live",
            timeout=TIMEOUT,
        )
        return r.status_code < 400, settings.minio_endpoint
    except Exception as exc:
        return False, str(exc)[:80]


def check_ollama() -> tuple[bool, str]:
    try:
        import httpx

        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=TIMEOUT)
        if r.status_code >= 400:
            return False, f"{settings.ollama_base_url} ({r.status_code})"
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_model = any(settings.llm_model.split(":")[0] in m for m in models)
        note = "model present" if has_model else f"MODEL '{settings.llm_model}' NOT PULLED"
        return True, f"{settings.ollama_base_url} ({note})"
    except Exception as exc:
        return False, str(exc)[:80]


REQUIRED = {
    "Postgres": check_postgres,
    "Redis": check_redis,
    "Qdrant": check_qdrant,
    "OpenSearch": check_opensearch,
    "MinIO": check_minio,
}
OPTIONAL = {
    "Ollama": check_ollama,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="BioIntel service healthcheck.")
    parser.add_argument(
        "--require-ollama",
        action="store_true",
        help="Treat Ollama as required (fail if unreachable).",
    )
    args = parser.parse_args()

    checks = list(REQUIRED.items()) + list(OPTIONAL.items())
    name_w = max(len(n) for n, _ in checks)

    print("\nBioIntel service healthcheck")
    print("=" * (name_w + 50))

    required_ok = True
    for name, fn in checks:
        ok, detail = fn()
        mark = "OK " if ok else "DOWN"
        optional = name in OPTIONAL
        tag = "  (optional)" if optional and not ok else ""
        print(f"  [{mark}] {name.ljust(name_w)}  {detail}{tag}")
        if not ok and (not optional or (name == "Ollama" and args.require_ollama)):
            required_ok = False

    print("=" * (name_w + 50))
    if required_ok:
        print("All required services reachable.\n")
        return 0
    print("One or more required services are DOWN. Run `make up` and retry.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
