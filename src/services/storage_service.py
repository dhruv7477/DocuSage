"""
src/services/storage_service.py
─────────────────────────────────
File storage service: any S3-compatible server or Amazon S3 behind one interface.

Storage path convention:
    workspaces/{workspace_id}/documents/{user_id}/{document_id}/{filename}

This path structure enforces isolation at the storage layer too —
a third layer of defence alongside the API auth and vector DB filters.

Reliability
-----------
- Uploads, downloads, and deletes retry transient connection errors
  (3 attempts, exponential backoff 200ms → 1s → 5s) via `tenacity`.
- A bounded connect/read timeout prevents the previous failure mode
  where a missing MinIO container caused requests to hang for minutes.
- `StorageConnectionError` is raised when the storage backend itself
  is unreachable, distinct from `StorageError` for terminal failures.
  Callers can use this to decide whether to retry at the task level.
"""

import asyncio
import io
from pathlib import Path

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.configs.config import get_settings
from src.core.exceptions import StorageError
from src.core.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class StorageConnectionError(StorageError):
    """
    Storage backend was unreachable (DNS, TCP, TLS handshake). Distinct
    from StorageError so background tasks can choose to retry these
    transient failures instead of marking a document FAILED.
    """

    error_code = "STORAGE_UNAVAILABLE"


# Maximum wall-clock retries inside the storage layer itself. Celery
# adds another retry tier on top for ingestion-time failures.
_MAX_ATTEMPTS = 3
_CONNECT_TIMEOUT_S = 30.0
_READ_TIMEOUT_S = 60.0


def _s3_compatible() -> bool:
    """True for the generic S3 client (SeaweedFS, MinIO, ...), False for AWS S3."""
    return settings.storage_backend != "s3"


def _is_transient_error(exc: BaseException) -> bool:
    """True if the exception looks like a network blip we should retry."""
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    # urllib3 / minio
    if name in {
        "MaxRetryError",
        "NewConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ProtocolError",
    }:
        return True
    # botocore — endpoint and read timeouts surface as these
    if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
        return True
    return False


class _RetryableStorageError(Exception):
    """Internal marker so tenacity only retries connection-class errors."""


def _classify(exc: BaseException) -> Exception:
    """Wrap connection-class errors so retry policy can pick them up."""
    if _is_transient_error(exc):
        return _RetryableStorageError(str(exc))
    return exc if isinstance(exc, Exception) else Exception(str(exc))


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "storage_retry",
        attempt=retry_state.attempt_number,
        next_sleep=retry_state.next_action.sleep if retry_state.next_action else None,
        error=str(exc) if exc else None,
    )


class StorageService:
    """
    Wraps an S3-compatible server (minio-py client) or AWS S3 (boto3) for
    file operations. Switching is one env var change.
    All operations run in executor to avoid blocking the async event loop.
    """

    def __init__(self) -> None:
        self._client = None
        self._bucket: str | None = None

    def _get_client(self):
        """
        Lazy-init storage client with bounded timeouts. Thread-safe via GIL.

        Connection-class failures here surface as StorageConnectionError so
        the API layer can return a 503-style typed error instead of a
        generic 500 when the storage backend is just not running.
        """
        if self._client is not None:
            return self._client, self._bucket

        backend = settings.storage_backend

        if _s3_compatible():
            try:
                import urllib3
                from minio import Minio
            except ImportError as exc:
                raise StorageError(
                    "minio not installed. Run: pip install minio"
                ) from exc

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(
                    connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S
                ),
                retries=urllib3.Retry(total=0),  # tenacity owns retry policy
                cert_reqs="CERT_REQUIRED" if settings.s3_secure else "CERT_NONE",
            )

            client = Minio(
                endpoint=settings.s3_endpoint,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                secure=settings.s3_secure,
                http_client=http_client,
            )
            bucket = settings.s3_bucket
            try:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
            except Exception as exc:
                if _is_transient_error(exc):
                    raise StorageConnectionError(
                        f"Cannot reach the S3 server at {settings.s3_endpoint}.",
                        detail="Start it (`make dev-up` starts SeaweedFS) and retry.",
                    ) from exc
                raise StorageError("S3 bucket setup failed.", detail=str(exc)) from exc
            self._client = client
            self._bucket = bucket

        elif backend == "s3":
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError as exc:
                raise StorageError(
                    "boto3 not installed. Run: pip install boto3"
                ) from exc

            session = boto3.Session(
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )
            self._client = session.client(
                "s3",
                config=BotoConfig(
                    connect_timeout=_CONNECT_TIMEOUT_S,
                    read_timeout=_READ_TIMEOUT_S,
                    retries={"max_attempts": 1, "mode": "standard"},
                ),
            )
            self._bucket = settings.s3_bucket
        else:
            raise StorageError(f"Unsupported storage backend: {backend}")

        return self._client, self._bucket

    @staticmethod
    def build_path(
        workspace_id: str, user_id: str, document_id: str, filename: str
    ) -> str:
        """
        Deterministic storage path. Consistent path = easy reconstruction.
        Example: workspaces/ws-123/documents/usr-456/doc-789/report.pdf
        """
        safe_filename = Path(filename).name  # Strip any path traversal attempts
        return f"workspaces/{workspace_id}/documents/{user_id}/{document_id}/{safe_filename}"

    # ── Public async API ──────────────────────────────────────────

    async def upload(
        self,
        content: bytes,
        path: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        await self._with_retry(
            "upload",
            path,
            self._upload_sync,
            content,
            path,
            content_type,
        )
        log.debug("file_uploaded", path=path, size=len(content))
        return path

    async def download(self, path: str) -> bytes:
        return await self._with_retry("download", path, self._download_sync, path)

    async def delete(self, path: str) -> None:
        await self._with_retry("delete", path, self._delete_sync, path)
        log.debug("file_deleted", path=path)

    async def get_presigned_url(self, path: str, expires_seconds: int = 3600) -> str:
        return await self._with_retry(
            "presigned_url", path, self._presigned_sync, path, expires_seconds
        )

    # ── Retry plumbing ────────────────────────────────────────────

    async def _with_retry(self, op: str, path: str, fn, *args):
        """
        Run a sync storage operation in a thread, retry transient errors.
        Terminal errors are wrapped in StorageError; connection errors in
        StorageConnectionError so the caller can distinguish.
        """
        loop = asyncio.get_event_loop()

        async def run_once():
            try:
                return await loop.run_in_executor(None, fn, *args)
            except Exception as exc:
                # Re-raise as a retry-marker if it's transient, else as-is
                raise _classify(exc) from exc

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=wait_exponential(multiplier=0.2, min=0.2, max=5.0),
                retry=retry_if_exception_type(_RetryableStorageError),
                reraise=True,
                before_sleep=_log_retry,
            ):
                with attempt:
                    return await run_once()
        except _RetryableStorageError as exc:
            raise StorageConnectionError(
                f"Storage {op} failed after {_MAX_ATTEMPTS} attempts: '{path}'.",
                detail=str(exc),
            ) from exc
        except RetryError as exc:  # pragma: no cover — defensive
            raise StorageConnectionError(
                f"Storage {op} retry exhausted: '{path}'.", detail=str(exc)
            ) from exc
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Storage {op} failed for '{path}'.", detail=str(exc)
            ) from exc

    # ── Sync backend calls ────────────────────────────────────────

    def _upload_sync(self, content: bytes, path: str, content_type: str) -> None:
        client, bucket = self._get_client()
        if _s3_compatible():
            client.put_object(
                bucket_name=bucket,
                object_name=path,
                data=io.BytesIO(content),
                length=len(content),
                content_type=content_type,
            )
        else:  # s3
            client.put_object(
                Bucket=bucket,
                Key=path,
                Body=content,
                ContentType=content_type,
            )

    def _download_sync(self, path: str) -> bytes:
        client, bucket = self._get_client()
        if _s3_compatible():
            response = client.get_object(bucket, path)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        else:  # s3
            response = client.get_object(Bucket=bucket, Key=path)
            return response["Body"].read()

    def _delete_sync(self, path: str) -> None:
        client, bucket = self._get_client()
        if _s3_compatible():
            client.remove_object(bucket, path)
        else:
            client.delete_object(Bucket=bucket, Key=path)

    def _presigned_sync(self, path: str, expires: int) -> str:
        client, bucket = self._get_client()
        from datetime import timedelta

        if _s3_compatible():
            return client.presigned_get_object(
                bucket, path, expires=timedelta(seconds=expires)
            )
        else:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": path},
                ExpiresIn=expires,
            )
