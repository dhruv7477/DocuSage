"""
tests/integration/test_demo_samples.py
──────────────────────────────────────
Demo mode end to end, below the API: the sample library is ingested once
through the real pipeline, a new account gets its own copies, deleting a
copy keeps the shared file, quotas ignore samples, and the nightly wipe
removes uploads but keeps samples.

Real Postgres and real Qdrant. Storage is in memory and the embedder is a
deterministic fake, so no model download is needed.
"""

import hashlib
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import fitz
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from src.configs.config import get_settings
from src.demo import samples
from src.demo.samples import (
    USER_WORKSPACE,
    SeedOutcome,
    ensure_library,
    seed_user,
    wipe_user_uploads,
)
from src.models import Document, User, Workspace
from src.models.document import DocumentStatus, DocumentType
from src.services.document_service import DocumentService
from src.services.quota_service import QuotaService
from src.vector_store import qdrant_store
from src.vector_store.base import SearchFilter
from src.vector_store.qdrant_store import QdrantVectorStore
from src.workers import runtime
from src.workers.tasks import ingest_document_task
from tests.fakes import InMemoryStorage

DIM = 16
CANARY = "MARIGOLD-2207"


def _reachable(url: str, default_port: int) -> bool:
    parts = urlparse(url if "://" in url else f"tcp://{url}")
    try:
        with socket.create_connection(
            (parts.hostname, parts.port or default_port), timeout=2
        ):
            return True
    except OSError:
        return False


_s = get_settings()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            _reachable(f"tcp://{_s.postgres_host}:{_s.postgres_port}", 5432)
            and _reachable(_s.qdrant_url, 6333)
        ),
        reason="needs a live Postgres and Qdrant",
    ),
]


class FakeEmbedder:
    """Deterministic vectors from text hashes; enough for storage and search."""

    dimension = DIM
    provider_name = "fake"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text_: str) -> list[float]:
        digest = hashlib.sha256(text_.encode()).digest()
        return [1.0 + b / 255 for b in digest[:DIM]]


def _make_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 760), body, fontsize=11)
    doc.save(path)
    doc.close()


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """Demo mode with two small sample PDFs, a private library and Qdrant prefix."""
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    _make_pdf(
        sample_dir / "handbook.pdf",
        [f"Handbook page one. The project codename is {CANARY}.", "Handbook page two."],
    )
    _make_pdf(sample_dir / "policy.pdf", ["Policy page one. Expenses need receipts."])

    settings = get_settings()
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "demo_samples_dir", str(sample_dir))
    monkeypatch.setattr(
        qdrant_store.settings,
        "qdrant_collection_prefix",
        f"demotest{uuid.uuid4().hex[:6]}",
    )
    monkeypatch.setattr(
        samples, "LIBRARY_EMAIL", f"library-{uuid.uuid4().hex[:8]}@docusage.invalid"
    )

    storage = InMemoryStorage()
    store = QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    with (
        patch("src.services.storage_service.StorageService", return_value=storage),
        patch(
            "src.providers.embedding.factory.get_embedding_provider",
            return_value=FakeEmbedder(),
        ),
        patch("src.vector_store.factory.get_vector_store", return_value=store),
    ):
        yield {
            "storage": storage,
            "store": store,
            "engine": create_engine(settings.database_url_sync),
        }


def _run(coro_factory):
    """Run `coro_factory(db)` on the worker loop with a fresh session."""

    async def go():
        async with runtime.session_factory()() as db:
            return await coro_factory(db)

    return runtime.run(go())


def _build_library(env: dict) -> None:
    async def ensure(db):
        ids = await ensure_library(db, env["storage"])
        await db.commit()
        return ids

    ids = _run(ensure)
    assert len(ids) == 2
    for doc_id in ids:
        assert ingest_document_task.apply(args=[doc_id]).successful()


def _new_user(engine) -> uuid.UUID:
    with Session(engine) as s:
        user = User(email=f"demo-{uuid.uuid4().hex[:10]}@example.com", full_name="Demo")
        s.add(user)
        s.commit()
        return user.id


def _user_docs(engine, user_id: uuid.UUID) -> list[Document]:
    with Session(engine) as s:
        return list(s.scalars(select(Document).where(Document.user_id == user_id)))


# ── Tests ─────────────────────────────────────────────────────────


def test_seeding_waits_until_the_library_is_ingested(demo_env: dict) -> None:
    user_id = _new_user(demo_env["engine"])

    assert _run(lambda db: seed_user(db, demo_env["store"], user_id)) == (
        SeedOutcome.LIBRARY_NOT_READY
    )


def test_new_account_gets_searchable_copies_of_every_sample(demo_env: dict) -> None:
    _build_library(demo_env)
    user_id = _new_user(demo_env["engine"])

    first = _run(lambda db: seed_user(db, demo_env["store"], user_id))
    again = _run(lambda db: seed_user(db, demo_env["store"], user_id))

    assert (first, again) == (SeedOutcome.SEEDED, SeedOutcome.ALREADY_SEEDED)
    docs = _user_docs(demo_env["engine"], user_id)
    assert sorted(d.original_filename for d in docs) == ["handbook.pdf", "policy.pdf"]
    assert {d.status for d in docs} == {DocumentStatus.COMPLETED}
    assert all(d.chunk_count > 0 for d in docs)

    ws_id = str(docs[0].workspace_id)
    with Session(demo_env["engine"]) as s:
        assert s.get(Workspace, docs[0].workspace_id).name == USER_WORKSPACE
    hits = runtime.run(
        demo_env["store"].text_search(ws_id, CANARY, SearchFilter(str(user_id), ws_id))
    )
    assert hits and all(h.payload.user_id == str(user_id) for h in hits)
    handbook = next(d for d in docs if d.original_filename == "handbook.pdf")
    chunks = runtime.run(
        demo_env["store"].list_document_chunks(ws_id, str(handbook.id), str(user_id))
    )
    chunk_ids = {c.chunk_id for c in chunks}
    for c in chunks:  # neighbour links point at the copies, not the library
        for link in (c.payload.prev_chunk_id, c.payload.next_chunk_id):
            assert link is None or link in chunk_ids


def test_samples_do_not_count_toward_quotas(demo_env: dict) -> None:
    _build_library(demo_env)
    user_id = _new_user(demo_env["engine"])
    _run(lambda db: seed_user(db, demo_env["store"], user_id))

    async def usage(db):
        user = await db.get(User, user_id)
        return await QuotaService(db)._document_usage(user)

    assert _run(usage) == (0, 0)


def test_deleting_a_sample_copy_keeps_the_shared_file(demo_env: dict) -> None:
    _build_library(demo_env)
    user_id = _new_user(demo_env["engine"])
    _run(lambda db: seed_user(db, demo_env["store"], user_id))
    copy = _user_docs(demo_env["engine"], user_id)[0]

    async def delete(db):
        user = await db.get(User, user_id)
        await DocumentService(db, demo_env["storage"]).delete_document(copy.id, user)
        await db.commit()

    _run(delete)

    assert copy.storage_path in demo_env["storage"].objects
    assert len(_user_docs(demo_env["engine"], user_id)) == 1


def test_nightly_wipe_removes_uploads_and_keeps_samples(demo_env: dict) -> None:
    _build_library(demo_env)
    user_id = _new_user(demo_env["engine"])
    _run(lambda db: seed_user(db, demo_env["store"], user_id))
    sample_ws = _user_docs(demo_env["engine"], user_id)[0].workspace_id
    upload_path = f"workspaces/{sample_ws}/documents/{user_id}/upload/mine.txt"
    demo_env["storage"].objects[upload_path] = b"my private upload"
    with Session(demo_env["engine"]) as s:
        s.add(
            Document(
                workspace_id=sample_ws,
                user_id=user_id,
                filename="mine.txt",
                original_filename="mine.txt",
                file_type=DocumentType.TXT,
                file_size_bytes=17,
                storage_path=upload_path,
                status=DocumentStatus.COMPLETED,
            )
        )
        s.commit()

    deleted = _run(
        lambda db: wipe_user_uploads(db, demo_env["storage"], demo_env["store"])
    )

    assert deleted >= 1
    remaining = _user_docs(demo_env["engine"], user_id)
    assert sorted(d.original_filename for d in remaining) == [
        "handbook.pdf",
        "policy.pdf",
    ]
    assert upload_path not in demo_env["storage"].objects
    with demo_env["engine"].connect() as conn:
        library_docs = conn.execute(
            text(
                "SELECT count(*) FROM documents d JOIN users u ON u.id = d.user_id "
                "WHERE u.email = :email"
            ),
            {"email": samples.LIBRARY_EMAIL},
        ).scalar_one()
    assert library_docs == 2


def test_library_owner_cannot_sign_in(demo_env: dict) -> None:
    _run(lambda db: ensure_library(db, demo_env["storage"]))

    with Session(demo_env["engine"]) as s:
        owner = s.scalar(select(User).where(User.email == samples.LIBRARY_EMAIL))
    assert owner is None or owner.is_active is False


def test_demo_tasks_do_nothing_outside_demo_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.workers.demo_tasks import (
        ensure_sample_library_task,
        seed_sample_documents_task,
        wipe_user_uploads_task,
    )

    monkeypatch.setattr(get_settings(), "demo_mode", False)
    with patch("src.workers.demo_tasks.runtime", MagicMock()) as rt:
        assert ensure_sample_library_task.apply().get() == 0
        assert (
            seed_sample_documents_task.apply(args=[str(uuid.uuid4())]).get()
            == "disabled"
        )
        assert wipe_user_uploads_task.apply().get() == 0
    rt.run.assert_not_called()
