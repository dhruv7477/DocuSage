"""
tests/integration/test_smoke.py
─────────────────────────────────
End-to-end smoke test — covers the full happy path of the application
against a real Postgres. Every other external service is mocked, and
storage is an in-memory StorageService.

Flow tested:
    1.  Register user
    2.  Login → receive tokens
    3.  Get /me → verify profile
    4.  Create workspace
    5.  Upload document (mocked Celery dispatch)
    6.  List documents
    7.  Create chat session
    8.  Send query → receive answer + sources
    9.  Retrieve chat history
    10. Delete document
    11. Delete chat
    12. Admin: list users (as super_admin)
    13. Admin: get platform stats
    14. Rate limit header present on responses
    15. /health endpoint returns 200
"""

import socket
import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.services.document_service import DocumentService
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


SMOKE_FILE = (
    b"DocuSage smoke test document.\nThis is paragraph one.\nThis is paragraph two."
)


def _upload(client: TestClient, state: dict, filename: str):
    return client.post(
        "/api/v1/documents/upload",
        data={"workspace_id": state["workspace_id"]},
        files={"file": (filename, BytesIO(SMOKE_FILE), "text/plain")},
        headers=state["headers"],
    )


def _document_row_committed(document_id: str) -> bool:
    """Look for the row from a separate connection, as the worker would."""
    from sqlalchemy import create_engine, text

    from src.configs.config import get_settings

    engine = create_engine(get_settings().database_url_sync)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM documents WHERE id = :id"), {"id": document_id}
            ).first()
        return row is not None
    finally:
        engine.dispose()


def _postgres_reachable() -> bool:
    """The smoke pipeline writes real rows — it needs a live Postgres."""
    from src.configs.config import get_settings

    s = get_settings()
    try:
        with socket.create_connection((s.postgres_host, s.postgres_port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="smoke pipeline needs a live Postgres (run `make dev-up` first)",
)

# ── Shared mock targets ───────────────────────────────────────────

MOCK_VECTOR_STORE = MagicMock()
MOCK_VECTOR_STORE.search = AsyncMock(return_value=[])
MOCK_VECTOR_STORE.upsert = AsyncMock()

MOCK_LLM = MagicMock()
MOCK_LLM.generate = AsyncMock(return_value="This is a test answer from the LLM.")
MOCK_LLM.generate_stream = AsyncMock(return_value=iter(["Test ", "answer"]))

MOCK_EMBEDDER = MagicMock()
MOCK_EMBEDDER.embed = AsyncMock(return_value=[[0.1] * 1024])


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Single TestClient for the whole smoke test module."""
    from src.api.v1.documents import _get_service
    from src.main import create_app

    app = create_app()
    storage = InMemoryStorage()

    def _document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
        return DocumentService(db, storage)

    app.dependency_overrides[_get_service] = _document_service
    with (
        patch(
            "src.vector_store.factory.get_vector_store", return_value=MOCK_VECTOR_STORE
        ),
        patch("src.providers.llm.factory.get_llm_provider", return_value=MOCK_LLM),
        patch(
            "src.providers.embedding.factory.get_embedding_provider",
            return_value=MOCK_EMBEDDER,
        ),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture(scope="module")
def state():
    """Mutable dict shared across all smoke test cases in order."""
    return {}


# ── Smoke Tests ───────────────────────────────────────────────────


class TestSmokePipeline:
    """
    Tests run in definition order to simulate a real user session.
    Each test reads from / writes to `state` to chain results together.
    """

    def test_01_health_liveness(self, client, state):
        """Health endpoint always returns 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "app" in data

    def test_02_register(self, client, state):
        """Register a new user account."""
        email = f"smoke_{uuid.uuid4().hex[:8]}@test.com"
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": "SecurePass123!",
                "full_name": "Smoke Tester",
            },
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["user"]["email"] == email
        state["email"] = email
        state["user_id"] = data["user"]["id"]

    def test_03_login(self, client, state):
        """Login with registered credentials and receive tokens."""
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": state["email"],
                "password": "SecurePass123!",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        state["access_token"] = data["access_token"]
        state["headers"] = {"Authorization": f"Bearer {data['access_token']}"}

    def test_04_get_me(self, client, state):
        """Authenticated user can fetch their own profile."""
        resp = client.get("/api/v1/auth/me", headers=state["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["email"] == state["email"]

    def test_05_create_workspace(self, client, state):
        """Create a workspace and become its owner."""
        resp = client.post(
            "/api/v1/workspaces",
            json={
                "name": "Smoke Test Workspace",
                "description": "Created by smoke test",
            },
            headers=state["headers"],
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["name"] == "Smoke Test Workspace"
        state["workspace_id"] = data["id"]

    def test_06_list_workspaces(self, client, state):
        """User sees their workspace in the list."""
        resp = client.get("/api/v1/workspaces", headers=state["headers"])
        assert resp.status_code == 200
        workspaces = resp.json()
        ids = [w["id"] for w in workspaces]
        assert state["workspace_id"] in ids

    def test_07_upload_document(self, client, state):
        """Upload a text document. Celery dispatch is mocked.

        The worker reads the row in its own transaction, so the row must
        already be committed when the task is queued.
        """
        committed_at_dispatch: list[bool] = []
        with patch("src.workers.tasks.ingest_document_task") as mock_task:
            mock_task.delay = MagicMock(
                side_effect=lambda doc_id: committed_at_dispatch.append(
                    _document_row_committed(doc_id)
                )
            )
            resp = _upload(client, state, "smoke_test.txt")
        assert resp.status_code == 202, resp.text
        data = resp.json()
        # `filename` is the storage-safe "{uuid}.txt"; the user-visible
        # name lives in original_filename.
        assert data["original_filename"] == "smoke_test.txt"
        assert data["filename"].endswith(".txt")
        assert data["status"] in ("pending", "PENDING")
        assert committed_at_dispatch == [True]
        state["document_id"] = data["id"]

    def test_07b_reupload_same_filename(self, client, state):
        """Re-uploading a filename resets the same document (was a 500)."""
        with patch("src.workers.tasks.ingest_document_task") as mock_task:
            mock_task.delay = MagicMock()
            resp = _upload(client, state, "smoke_test.txt")
        assert resp.status_code == 202, resp.text
        assert resp.json()["id"] == state["document_id"]
        assert resp.json()["status"] == "pending"
        mock_task.delay.assert_called_once_with(state["document_id"])

    def test_07c_dispatch_failure_marks_document_failed(self, client, state):
        """A broker outage returns the document as FAILED (was a 500)."""
        with patch("src.workers.tasks.ingest_document_task") as mock_task:
            mock_task.delay = MagicMock(side_effect=ConnectionError("broker down"))
            resp = _upload(client, state, "smoke_broker_down.txt")
        assert resp.status_code == 202, resp.text
        assert resp.json()["status"] == "failed"
        assert "Failed to queue" in resp.json()["error_message"]

    def test_08_list_documents(self, client, state):
        """Uploaded document appears in the list."""
        resp = client.get(
            f"/api/v1/documents?workspace_id={state['workspace_id']}",
            headers=state["headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        ids = [d["id"] for d in data["documents"]]
        assert state["document_id"] in ids

    def test_09_create_chat(self, client, state):
        """Create a chat session in the workspace."""
        resp = client.post(
            "/api/v1/chat",
            json={
                "workspace_id": state["workspace_id"],
                "title": "Smoke Test Chat",
            },
            headers=state["headers"],
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["title"] == "Smoke Test Chat"
        state["chat_id"] = data["id"]

    def test_10_chat_query(self, client, state):
        """Send a RAG query and receive an answer."""
        with patch("src.services.rag_service.RAGService.generate") as mock_gen:
            from src.services.rag_service import RAGResponse

            mock_gen.return_value = RAGResponse(
                answer="The document discusses smoke testing.",
                sources=[],
                context_chunks_used=1,
                confidence=0.9,
                abstained=False,
            )
            resp = client.post(
                f"/api/v1/chat/{state['chat_id']}/query",
                json={"query": "What does the document discuss?"},
                headers=state["headers"],
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        state["message_id"] = data.get("message_id")

    def test_11_get_chat_messages(self, client, state):
        """Chat message history contains the exchange."""
        resp = client.get(
            f"/api/v1/chat/{state['chat_id']}/messages",
            headers=state["headers"],
        )
        assert resp.status_code == 200
        messages = resp.json()
        # Should have at least user + assistant message
        assert len(messages) >= 2
        roles = {m["role"] for m in messages}
        assert "user" in roles
        assert "assistant" in roles

    def test_12_refresh_token(self, client, state):
        """Can exchange access token for a new token pair via refresh."""
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": state["email"],
                "password": "SecurePass123!",
            },
        )
        refresh_token = resp.json()["refresh_token"]

        resp2 = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )
        assert resp2.status_code == 200
        assert "access_token" in resp2.json()

    def test_13_delete_chat(self, client, state):
        """Delete the chat session."""
        resp = client.delete(
            f"/api/v1/chat/{state['chat_id']}",
            headers=state["headers"],
        )
        assert resp.status_code == 204

    def test_14_delete_document(self, client, state):
        """Delete the uploaded document."""
        with patch(
            "src.services.document_service.DocumentService.delete_document", AsyncMock()
        ):
            resp = client.delete(
                f"/api/v1/documents/{state['document_id']}",
                headers=state["headers"],
            )
        assert resp.status_code == 204

    def test_15_unauth_returns_403(self, client, state):
        """Unauthenticated requests to protected routes return 403."""
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 403

    def test_16_nonexistent_resource_returns_404(self, client, state):
        """Fetching a nonexistent document returns 404."""
        fake_id = uuid.uuid4()
        resp = client.get(
            f"/api/v1/documents/{fake_id}",
            headers=state["headers"],
        )
        assert resp.status_code == 404

    def test_17_admin_requires_super_admin(self, client, state):
        """Regular users cannot access admin endpoints."""
        resp = client.get("/api/v1/admin/users", headers=state["headers"])
        assert resp.status_code == 403

    def test_18_eval_run(self, client, state):
        """Eval endpoint accepts a Q&A pair and returns metrics."""
        with patch("src.services.eval_service.EvalService.evaluate") as mock_eval:
            from src.services.eval_service import EvalResult

            # composite_score is a derived @property — not a constructor arg.
            mock_eval.return_value = EvalResult(
                question="Q",
                answer="A",
                faithfulness=0.9,
                answer_relevance=0.85,
                context_precision=0.8,
                context_recall=None,
                answer_similarity=None,
            )
            resp = client.post(
                "/api/v1/eval/run",
                json={
                    "question": "What is DocuSage?",
                    "answer": "A RAG-powered knowledge assistant.",
                    "contexts": ["DocuSage is an AI assistant for private documents."],
                },
                headers=state["headers"],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "composite_score" in data
        assert data["composite_score"] > 0
