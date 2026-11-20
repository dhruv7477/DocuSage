"""
tests/unit/test_qdrant_upsert_batches.py
────────────────────────────────────────
QdrantVectorStore sends upserts in UPSERT_BATCH-sized requests (Qdrant keeps
the memory it used to take in one large request) and keeps the result all
or nothing, as one request was. The client is a fake; no Qdrant runs.
"""

import pytest
from qdrant_client.http import models as qmodels

from src.vector_store import qdrant_store
from src.vector_store.qdrant_store import UPSERT_BATCH, QdrantVectorStore

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self, fail_on_call: int | None = None, fail_delete: bool = False):
        self.upserts: list[list] = []
        self.deleted: list = []
        self._fail_on_call = fail_on_call
        self._fail_delete = fail_delete

    async def upsert(self, collection_name, points, wait):
        assert wait is True
        if self._fail_on_call == len(self.upserts) + 1:
            raise RuntimeError("qdrant went away")
        self.upserts.append([p.id for p in points])

    async def delete(self, collection_name, points_selector, wait):
        if self._fail_delete:
            raise RuntimeError("still away")
        self.deleted.extend(points_selector.points)


def _store(client: _FakeClient) -> QdrantVectorStore:
    store = QdrantVectorStore.__new__(QdrantVectorStore)  # no real client
    store._client = client
    return store


def _points(n: int) -> list[qmodels.PointStruct]:
    return [qmodels.PointStruct(id=i, vector=[0.0, 1.0]) for i in range(n)]


async def test_sends_points_in_batches() -> None:
    client = _FakeClient()
    n = 2 * UPSERT_BATCH + 44

    await _store(client)._upsert_in_batches("c", _points(n))

    assert [len(b) for b in client.upserts] == [UPSERT_BATCH, UPSERT_BATCH, 44]
    assert [i for b in client.upserts for i in b] == list(range(n))
    assert client.deleted == []


async def test_a_small_document_is_one_request() -> None:
    client = _FakeClient()
    await _store(client)._upsert_in_batches("c", _points(5))
    assert client.upserts == [[0, 1, 2, 3, 4]]


async def test_a_failed_batch_removes_the_batches_already_written() -> None:
    client = _FakeClient(fail_on_call=3)

    with pytest.raises(RuntimeError, match="qdrant went away"):
        await _store(client)._upsert_in_batches("c", _points(3 * UPSERT_BATCH))

    assert client.deleted == list(range(2 * UPSERT_BATCH))


async def test_a_failed_first_batch_deletes_nothing() -> None:
    client = _FakeClient(fail_on_call=1)
    with pytest.raises(RuntimeError):
        await _store(client)._upsert_in_batches("c", _points(10))
    assert client.deleted == []


async def test_a_failed_cleanup_is_logged_and_the_original_error_raised(
    monkeypatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        qdrant_store.log, "warning", lambda event, **_kw: warnings.append(event)
    )
    client = _FakeClient(fail_on_call=2, fail_delete=True)

    with pytest.raises(RuntimeError, match="qdrant went away"):
        await _store(client)._upsert_in_batches("c", _points(2 * UPSERT_BATCH))

    assert warnings == ["partial_upsert_cleanup_failed"]
