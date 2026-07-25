"""MinIO (S3-compatible) object storage client for raw source payloads.

Raw API responses / documents are archived here so ingestion is auditable and
re-processable without re-hitting external APIs.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from biointel.common.config import settings
from biointel.common.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_minio() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket(bucket: str | None = None) -> str:
    bucket = bucket or settings.minio_bucket
    client = get_minio()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("Created MinIO bucket: %s", bucket)
    except S3Error as exc:  # pragma: no cover - network dependent
        logger.error("MinIO bucket error: %s", exc)
        raise
    return bucket


def put_raw(
    key: str, data: bytes | dict | str, content_type: str = "application/octet-stream"
) -> str:
    """Store a raw payload and return the object key."""
    bucket = ensure_bucket()
    if isinstance(data, dict):
        payload = json.dumps(data).encode("utf-8")
        content_type = "application/json"
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data
    get_minio().put_object(
        bucket, key, io.BytesIO(payload), length=len(payload), content_type=content_type
    )
    return key


def get_raw(key: str, bucket: str | None = None) -> bytes:
    bucket = bucket or settings.minio_bucket
    resp = get_minio().get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()
