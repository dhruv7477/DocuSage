"""
tests/integration/test_storage_s3.py
────────────────────────────────────
StorageService against a real S3-compatible server: SeaweedFS, which
replaces MinIO in both compose files. CI starts it from
docker/seaweedfs/start.sh; locally run the dev compose `seaweedfs`
service and point S3_ENDPOINT at it.
"""

import socket
import urllib.request
import uuid

import pytest

from src.core.exceptions import StorageError
from src.services import storage_service
from src.services.storage_service import StorageConnectionError, StorageService


def _storage_reachable() -> bool:
    host, _, port = storage_service.settings.s3_endpoint.partition(":")
    try:
        with socket.create_connection((host, int(port or 80)), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        storage_service.settings.storage_backend == "s3" or not _storage_reachable(),
        reason="needs an S3-compatible server at S3_ENDPOINT (SeaweedFS)",
    ),
]


def _path() -> str:
    return f"workspaces/test/documents/u/{uuid.uuid4()}/file.pdf"


async def test_round_trip_presigned_url_and_delete() -> None:
    storage = StorageService()
    path = _path()
    body = bytes(range(256)) * 1024  # binary, 256 KiB

    await storage.upload(body, path, "application/pdf")
    assert await storage.download(path) == body

    url = await storage.get_presigned_url(path, expires_seconds=60)
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (test URL)
        assert resp.read() == body

    await storage.delete(path)
    with pytest.raises(StorageError):
        await storage.download(path)


async def test_wrong_secret_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_service.settings, "s3_secret_key", "wrong-secret")
    storage = StorageService()

    with pytest.raises(StorageError) as exc_info:
        await storage.upload(b"x", _path(), "text/plain")

    assert not isinstance(exc_info.value, StorageConnectionError)


async def test_unreachable_endpoint_is_a_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage_service.settings, "s3_endpoint", "127.0.0.1:1")
    storage = StorageService()

    with pytest.raises(StorageConnectionError):
        await storage.upload(b"x", _path(), "text/plain")
