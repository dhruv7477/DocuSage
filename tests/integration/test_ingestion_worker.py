"""
tests/integration/test_ingestion_worker.py
──────────────────────────────────────────
Ingestion tasks run the way a Celery worker process runs them, against a
real Postgres. Storage, embeddings and the vector store are mocked.

Regressions covered:
- Every second ingestion in one worker process failed with
  "Event loop is closed" (asyncpg pool reused across asyncio.run loops).
- Documents stayed PENDING forever after an unhandled task error.
- Stage progress was flushed but never committed, so the API never saw it.
- Nothing failed documents that were stuck in flight.
"""

import socket
import uuid
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine, create_engine, delete, text
from sqlalchemy.orm import Session

from src.configs.config import get_settings
from src.models import Document, User, Workspace
from src.models.document import DocumentStatus, DocumentType
from src.workers import runtime
from src.workers.tasks import (
    FAILED_AFTER_RETRIES_MESSAGE,
    STALE_MESSAGE,
    fail_stale_documents_task,
    ingest_document_task,
    mark_document_failed,
)

EMBEDDING_DIM = 384
DOCUMENT_TEXT = (
    b"DocuSage worker regression document.\n\n"
    b"Expenses above 500 euros need a manager's approval.\n\n"
    b"Remote work is allowed up to three days a week.\n"
)


def _postgres_reachable() -> bool:
    s = get_settings()
    try:
        with socket.create_connection((s.postgres_host, s.postgres_port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="worker tests need a live Postgres (run `make dev-up` first)",
    ),
]


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sync_engine() -> Iterator[Engine]:
    """Plain psycopg2 engine for test setup and checks, separate from the worker's."""
    engine = create_engine(get_settings().database_url_sync)
    yield engine
    engine.dispose()


@pytest.fixture
def owner(sync_engine: Engine) -> Iterator[tuple[uuid.UUID, uuid.UUID]]:
    """A user and their workspace, removed with their documents afterwards."""
    with Session(sync_engine) as s:
        user = User(
            email=f"worker-{uuid.uuid4().hex[:10]}@example.com",
            full_name="Worker Test",
        )
        s.add(user)
        s.flush()
        workspace = Workspace(name="worker-test", owner_id=user.id)
        s.add(workspace)
        s.commit()
        ids = (user.id, workspace.id)
    yield ids
    with Session(sync_engine) as s:
        s.execute(delete(Document).where(Document.user_id == ids[0]))
        s.execute(delete(Workspace).where(Workspace.id == ids[1]))
        s.execute(delete(User).where(User.id == ids[0]))
        s.commit()


@pytest.fixture
def pipeline_mocks() -> Iterator[MagicMock]:
    """Patch storage, embedder and vector store; return the embedder mock."""
    storage = MagicMock()
    storage.download = AsyncMock(return_value=DOCUMENT_TEXT)

    embedder = MagicMock()
    embedder.dimension = EMBEDDING_DIM
    embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
    )

    vector_store = MagicMock()
    vector_store.collection_exists = AsyncMock(return_value=True)
    vector_store.delete_document_vectors = AsyncMock()
    vector_store.upsert_vectors = AsyncMock()

    with (
        patch("src.services.storage_service.StorageService", return_value=storage),
        patch(
            "src.providers.embedding.factory.get_embedding_provider",
            return_value=embedder,
        ),
        patch("src.vector_store.factory.get_vector_store", return_value=vector_store),
    ):
        yield embedder


# ── Helpers ───────────────────────────────────────────────────────


def _add_document(
    engine: Engine,
    owner: tuple[uuid.UUID, uuid.UUID],
    status: DocumentStatus = DocumentStatus.PENDING,
) -> str:
    user_id, workspace_id = owner
    with Session(engine) as s:
        doc = Document(
            workspace_id=workspace_id,
            user_id=user_id,
            filename=f"{uuid.uuid4()}.txt",
            original_filename="notes.txt",
            file_type=DocumentType.TXT,
            file_size_bytes=len(DOCUMENT_TEXT),
            storage_path=f"workspaces/{workspace_id}/documents/{user_id}/notes.txt",
            status=status,
        )
        s.add(doc)
        s.commit()
        return str(doc.id)


def _row(engine: Engine, document_id: str):
    with engine.connect() as c:
        return c.execute(
            text(
                "SELECT status, chunk_count, error_message FROM documents WHERE id = :id"
            ),
            {"id": document_id},
        ).one()


# ── Tests ─────────────────────────────────────────────────────────


def test_five_ingestions_in_one_worker_process_all_complete(
    sync_engine: Engine, owner, pipeline_mocks
) -> None:
    doc_ids = [_add_document(sync_engine, owner) for _ in range(5)]

    results = [ingest_document_task.apply(args=[doc_id]) for doc_id in doc_ids]

    for doc_id, result in zip(doc_ids, results, strict=True):
        assert result.successful(), f"{doc_id}: {result.result!r}"
        status, chunk_count, error = _row(sync_engine, doc_id)
        assert (status, error) == ("completed", None)
        assert chunk_count > 0


def test_stage_progress_is_committed_while_the_task_runs(
    sync_engine: Engine, owner, pipeline_mocks
) -> None:
    doc_id = _add_document(sync_engine, owner)
    seen: list[str] = []

    async def embed_and_look(texts: list[str]) -> list[list[float]]:
        # Read from a separate connection, as the API would.
        seen.append(_row(sync_engine, doc_id).status)
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    pipeline_mocks.embed_documents.side_effect = embed_and_look

    assert ingest_document_task.apply(args=[doc_id]).successful()
    assert seen == ["embedding"]


@pytest.mark.parametrize(
    "error",
    [RuntimeError("unexpected"), ConnectionError("broker or storage down")],
    ids=["unhandled-error", "transient-error-retries-exhausted"],
)
def test_failed_task_marks_document_failed(
    sync_engine: Engine, owner, error: Exception
) -> None:
    doc_id = _add_document(sync_engine, owner)

    with patch("src.workers.tasks._run_ingestion", side_effect=error):
        result = ingest_document_task.apply(args=[doc_id])

    assert result.failed()
    status, _, message = _row(sync_engine, doc_id)
    assert (status, message) == ("failed", FAILED_AFTER_RETRIES_MESSAGE)


def test_mark_failed_leaves_finished_documents_alone(
    sync_engine: Engine, owner
) -> None:
    doc_id = _add_document(sync_engine, owner, status=DocumentStatus.COMPLETED)

    changed = runtime.run(mark_document_failed(doc_id, "should not apply"))

    assert changed is False
    assert _row(sync_engine, doc_id).status == "completed"


def test_stale_sweep_fails_only_stuck_in_flight_documents(
    sync_engine: Engine, owner
) -> None:
    stuck = _add_document(sync_engine, owner, status=DocumentStatus.EMBEDDING)
    fresh = _add_document(sync_engine, owner, status=DocumentStatus.PENDING)
    old_done = _add_document(sync_engine, owner, status=DocumentStatus.COMPLETED)
    with sync_engine.begin() as c:
        c.execute(
            text(
                "UPDATE documents SET updated_at = now() - interval '3 hours' "
                "WHERE id IN (:stuck, :old_done)"
            ),
            {"stuck": stuck, "old_done": old_done},
        )

    count = fail_stale_documents_task.apply().get()

    assert count >= 1
    assert tuple(_row(sync_engine, stuck)) == ("failed", 0, STALE_MESSAGE)
    assert _row(sync_engine, fresh).status == "pending"
    assert _row(sync_engine, old_done).status == "completed"
