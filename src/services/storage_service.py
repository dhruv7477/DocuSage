"""
src/services/storage_service.py
─────────────────────────────────
File storage service — abstracts S3 and MinIO behind a common interface.

Storage path convention:
    workspaces/{workspace_id}/documents/{user_id}/{document_id}/{filename}

This path structure enforces isolation at the storage layer too —
a third layer of defence alongside the API auth and vector DB filters.
"""

import asyncio
import io
import uuid
from pathlib import Path

from src.configs.config import get_settings
from src.core.exceptions import StorageError
from src.core.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class StorageService:
    """
    Wraps MinIO (development) or AWS S3 (production) for file operations.
    Uses the same boto3/minio interface — switching is one env var change.
    All operations run in executor to avoid blocking the async event loop.
    """

    def __init__(self) -> None:
        self._client = None
        self._bucket = None

    def _get_client(self):
        """Lazy-init storage client. Thread-safe via GIL."""
        if self._client is not None:
            return self._client, self._bucket

        backend = settings.storage_backend

        if backend == "minio":
            try:
                from minio import Minio
            except ImportError as exc:
                raise StorageError("minio not installed. Run: pip install minio") from exc

            client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            bucket = settings.minio_bucket
            # Create bucket if it doesn't exist
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            self._client = client
            self._bucket = bucket

        elif backend == "s3":
            try:
                import boto3
            except ImportError as exc:
                raise StorageError("boto3 not installed. Run: pip install boto3") from exc

            session = boto3.Session(
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )
            self._client = session.client("s3")
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

    async def upload(
        self,
        content: bytes,
        path: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload bytes to storage. Returns the storage path.
        Runs synchronous client in thread pool.
        """
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self._upload_sync, content, path, content_type
            )
            log.debug("file_uploaded", path=path, size=len(content))
            return path
        except Exception as exc:
            raise StorageError(f"Upload failed for '{path}'", detail=str(exc)) from exc

    def _upload_sync(self, content: bytes, path: str, content_type: str) -> None:
        client, bucket = self._get_client()
        backend = settings.storage_backend

        if backend == "minio":
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

    async def download(self, path: str) -> bytes:
        """Download a file from storage and return its bytes."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._download_sync, path)
        except Exception as exc:
            raise StorageError(f"Download failed for '{path}'", detail=str(exc)) from exc

    def _download_sync(self, path: str) -> bytes:
        client, bucket = self._get_client()
        backend = settings.storage_backend

        if backend == "minio":
            response = client.get_object(bucket, path)
            data = response.read()
            response.close()
            return data
        else:  # s3
            response = client.get_object(Bucket=bucket, Key=path)
            return response["Body"].read()

    async def delete(self, path: str) -> None:
        """Delete a file from storage."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._delete_sync, path)
            log.debug("file_deleted", path=path)
        except Exception as exc:
            raise StorageError(f"Delete failed for '{path}'", detail=str(exc)) from exc

    def _delete_sync(self, path: str) -> None:
        client, bucket = self._get_client()
        backend = settings.storage_backend
        if backend == "minio":
            client.remove_object(bucket, path)
        else:
            client.delete_object(Bucket=bucket, Key=path)

    async def get_presigned_url(self, path: str, expires_seconds: int = 3600) -> str:
        """Generate a time-limited pre-signed URL for direct file download."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._presigned_sync, path, expires_seconds
            )
        except Exception as exc:
            raise StorageError(f"Presigned URL failed for '{path}'", detail=str(exc)) from exc

    def _presigned_sync(self, path: str, expires: int) -> str:
        client, bucket = self._get_client()
        backend = settings.storage_backend
        from datetime import timedelta
        if backend == "minio":
            return client.presigned_get_object(bucket, path, expires=timedelta(seconds=expires))
        else:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": path},
                ExpiresIn=expires,
            )
