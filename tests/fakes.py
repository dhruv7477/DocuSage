"""Test doubles shared by several test modules."""

from src.services.storage_service import StorageService


class InMemoryStorage(StorageService):
    """StorageService that keeps objects in a dict instead of an S3 bucket.

    Only the blocking client calls are replaced, so path building and the
    retry wrapper are the real code.
    """

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}

    def _upload_sync(self, content: bytes, path: str, content_type: str) -> None:
        self.objects[path] = content

    def _download_sync(self, path: str) -> bytes:
        return self.objects[path]

    def _delete_sync(self, path: str) -> None:
        self.objects.pop(path, None)

    def _presigned_sync(self, path: str, expires: int) -> str:
        return f"memory://{path}"
