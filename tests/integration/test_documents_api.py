"""
tests/integration/test_documents_api.py
─────────────────────────────────────────
Integration tests for /api/v1/documents/* endpoints.
"""

import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


def _make_mock_user(uid: uuid.UUID):
    from src.models.user import User, UserRole
    user = User()
    user.id = uid
    user.email = "alice@example.com"
    user.full_name = "Alice"
    user.role = UserRole.MEMBER
    user.is_active = True
    user.is_email_verified = True
    return user


def _make_auth_headers(user_id: uuid.UUID) -> dict:
    from src.configs.config import get_settings
    from src.core.security import create_access_token

    settings = get_settings()
    token = create_access_token(
        subject=str(user_id),
        secret_key=settings.secret_key,
        algorithm=settings.algorithm,
        expires_minutes=settings.access_token_expire_minutes,
    )
    return {"Authorization": f"Bearer {token}"}


@contextmanager
def _auth_as(test_app, mock_user):
    """
    Override the get_current_user dependency for the duration of a test.

    NOTE: `unittest.mock.patch("src.dependencies.get_current_user")` does
    NOT work here — FastAPI captured the original function object in
    `Depends(...)` at import time, so swapping the module attribute has
    no effect. `dependency_overrides` is the supported mechanism.
    """
    from src.dependencies import get_current_user
    test_app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        yield
    finally:
        test_app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def test_app():
    return create_app()


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def workspace_id():
    return uuid.uuid4()


@pytest.fixture
def headers(user_id):
    return _make_auth_headers(user_id)


class TestDocumentUpload:

    def test_upload_requires_auth(self, test_app):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.post("/api/v1/documents/upload", data={"workspace_id": str(uuid.uuid4())})
        assert resp.status_code in (401, 403)

    def test_upload_missing_file_returns_422(self, test_app, user_id, workspace_id, headers):
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            with _auth_as(test_app, mock_user):
                resp = c.post(
                    "/api/v1/documents/upload",
                    data={"workspace_id": str(workspace_id)},
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_upload_unsupported_filetype_returns_422(
        self, test_app, user_id, workspace_id, headers
    ):
        from src.core.exceptions import UnsupportedFileTypeError
        mock_user = _make_mock_user(user_id)

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.documents.DocumentService") as MockSvc:
                mock_svc = AsyncMock()
                mock_svc.upload_document.side_effect = UnsupportedFileTypeError(
                    ".exe files are not supported."
                )
                MockSvc.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.post(
                        "/api/v1/documents/upload",
                        data={"workspace_id": str(workspace_id)},
                        files={"file": ("test.exe", b"MZ\x90\x00", "application/octet-stream")},
                        headers=headers,
                    )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "UNSUPPORTED_FILE_TYPE"

    def test_upload_success_returns_202_pending(
        self, test_app, user_id, workspace_id, headers
    ):
        from datetime import datetime, timezone
        mock_user = _make_mock_user(user_id)

        # SimpleNamespace, not MagicMock: pydantic's from_attributes reads
        # every schema field, and MagicMock auto-creates garbage attributes
        # (e.g. `progress`) that fail validation with a 500.
        from types import SimpleNamespace
        from src.models.document import DocumentStatus, DocumentType
        now = datetime.now(timezone.utc)
        mock_doc = SimpleNamespace(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            user_id=user_id,
            filename="abc123.txt",
            original_filename="notes.txt",
            file_type=DocumentType.TXT,
            file_size_bytes=100,
            status=DocumentStatus.PENDING,
            progress=0,
            chunk_count=0,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.documents.DocumentService") as MockSvc:
                mock_svc = AsyncMock()
                mock_svc.upload_document.return_value = mock_doc
                MockSvc.return_value = mock_svc

                # The endpoint imports the task inside the function body, so
                # patch it at its real location (module attribute lookup at
                # call time makes this effective).
                with patch("src.workers.tasks.ingest_document_task"):
                    with _auth_as(test_app, mock_user):
                        resp = c.post(
                            "/api/v1/documents/upload",
                            data={"workspace_id": str(workspace_id)},
                            files={"file": ("notes.txt", b"Hello world " * 10, "text/plain")},
                            headers=headers,
                        )

        assert resp.status_code == 202
        data = resp.json()
        assert data["original_filename"] == "notes.txt"
        assert data["status"] == "pending"


class TestListDocuments:

    def test_list_requires_auth(self, test_app, workspace_id):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.get(f"/api/v1/documents?workspace_id={workspace_id}")
        assert resp.status_code in (401, 403)

    def test_list_returns_empty_array(self, test_app, user_id, workspace_id, headers):
        mock_user = _make_mock_user(user_id)

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.documents.DocumentService") as MockSvc:
                mock_svc = AsyncMock()
                mock_svc.list_documents.return_value = []
                MockSvc.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.get(
                        f"/api/v1/documents?workspace_id={workspace_id}",
                        headers=headers,
                    )
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total"] == 0


class TestGetDocument:

    def test_get_not_found_returns_404(self, test_app, user_id, headers):
        from src.core.exceptions import DocumentNotFoundError
        mock_user = _make_mock_user(user_id)
        doc_id = uuid.uuid4()

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.documents.DocumentService") as MockSvc:
                mock_svc = AsyncMock()
                mock_svc.get_document.side_effect = DocumentNotFoundError("Not found.")
                MockSvc.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.get(
                        f"/api/v1/documents/{doc_id}",
                        headers=headers,
                    )
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "DOCUMENT_NOT_FOUND"

    def test_get_wrong_user_returns_403(self, test_app, user_id, headers):
        from src.core.exceptions import DocumentAccessDeniedError
        mock_user = _make_mock_user(user_id)
        doc_id = uuid.uuid4()

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.documents.DocumentService") as MockSvc:
                mock_svc = AsyncMock()
                mock_svc.get_document.side_effect = DocumentAccessDeniedError(
                    "Access denied."
                )
                MockSvc.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.get(
                        f"/api/v1/documents/{doc_id}",
                        headers=headers,
                    )
        assert resp.status_code == 403
        assert resp.json()["error_code"] == "DOCUMENT_ACCESS_DENIED"
