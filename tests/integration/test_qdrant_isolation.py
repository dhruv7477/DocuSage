"""
tests/integration/test_qdrant_isolation.py
──────────────────────────────────────────
Tenant isolation in QdrantVectorStore, against a real Qdrant.

Layout for every test (fresh workspace IDs each time):
    W1: Alice's document and Bob's document (Bob is a member of W1)
    W2: Bob's second document

All points get almost the same vector, so a similarity search without
the isolation filters would return everything. Each read or delete
method must only see the caller's own points in the workspace asked for.
"""

import socket
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse

import pytest

from src.configs.config import get_settings
from src.vector_store import qdrant_store
from src.vector_store.base import SearchFilter, VectorPayload
from src.vector_store.qdrant_store import QdrantVectorStore

DIM = 8
ALICE_CANARY = "heliotrope"
BOB_CANARY = "zinnia"


def _qdrant_reachable() -> bool:
    url = urlparse(get_settings().qdrant_url)
    try:
        with socket.create_connection((url.hostname, url.port or 6333), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _qdrant_reachable(), reason="needs a live Qdrant at QDRANT_URL"
    ),
]


@dataclass
class Tenants:
    store: QdrantVectorStore
    w1: str
    w2: str
    alice: str
    bob: str
    alice_doc: str
    bob_doc: str
    bob_w2_doc: str
    alice_chunks: list[str]
    bob_chunks: list[str]


def _vector(i: int) -> list[float]:
    return [1.0] + [0.001 * i] * (DIM - 1)


async def _put(
    store: QdrantVectorStore, ws: str, user: str, doc: str, texts: list[str]
) -> list[str]:
    ids = [str(uuid.uuid4()) for _ in texts]
    payloads = [
        VectorPayload(
            user_id=user,
            workspace_id=ws,
            document_id=doc,
            chunk_index=i,
            text=t,
            source=f"{doc}.txt",
            chunk_total=len(texts),
        )
        for i, t in enumerate(texts)
    ]
    await store.upsert_vectors(
        ws, ids, [_vector(i) for i in range(len(texts))], payloads
    )
    return ids


@pytest.fixture
async def tenants(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Tenants]:
    monkeypatch.setattr(
        qdrant_store.settings,
        "qdrant_collection_prefix",
        f"isotest{uuid.uuid4().hex[:6]}",
    )
    s = get_settings()
    store = QdrantVectorStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    w1, w2 = str(uuid.uuid4()), str(uuid.uuid4())
    alice, bob = str(uuid.uuid4()), str(uuid.uuid4())
    alice_doc, bob_doc, bob_w2_doc = (str(uuid.uuid4()) for _ in range(3))
    for ws in (w1, w2):
        await store.create_collection(ws, dimension=DIM)

    alice_chunks = await _put(
        store,
        w1,
        alice,
        alice_doc,
        [f"Alice's plan: the codename is {ALICE_CANARY}.", "Alice's second chunk."],
    )
    bob_chunks = await _put(
        store,
        w1,
        bob,
        bob_doc,
        [f"Bob's notes mention {BOB_CANARY}.", "Bob's second chunk."],
    )
    await _put(store, w2, bob, bob_w2_doc, ["Bob's other workspace."])

    yield Tenants(
        store,
        w1,
        w2,
        alice,
        bob,
        alice_doc,
        bob_doc,
        bob_w2_doc,
        alice_chunks,
        bob_chunks,
    )
    for ws in (w1, w2):
        await store.delete_collection(ws)


def _users(results) -> set[str]:
    return {r.payload.user_id for r in results}


async def test_search_returns_only_the_callers_points(tenants: Tenants) -> None:
    t = tenants

    bob_in_w1 = await t.store.search(
        t.w1, _vector(0), SearchFilter(t.bob, t.w1), top_k=50
    )
    alice_in_w1 = await t.store.search(
        t.w1, _vector(0), SearchFilter(t.alice, t.w1), top_k=50
    )
    alice_in_w2 = await t.store.search(
        t.w2, _vector(0), SearchFilter(t.alice, t.w2), top_k=50
    )

    assert _users(bob_in_w1) == {t.bob}
    assert {r.payload.document_id for r in bob_in_w1} == {t.bob_doc}
    assert _users(alice_in_w1) == {t.alice}
    assert alice_in_w2 == []


async def test_search_ignores_a_filter_for_another_workspace(tenants: Tenants) -> None:
    t = tenants

    # Bob's W2 filter used against W1's collection matches nothing there.
    results = await t.store.search(
        t.w1, _vector(0), SearchFilter(t.bob, t.w2), top_k=50
    )

    assert results == []


async def test_text_search_never_matches_another_users_text(tenants: Tenants) -> None:
    t = tenants

    bob_looks = await t.store.text_search(t.w1, ALICE_CANARY, SearchFilter(t.bob, t.w1))
    alice_looks = await t.store.text_search(
        t.w1, ALICE_CANARY, SearchFilter(t.alice, t.w1)
    )

    assert bob_looks == []
    assert [ALICE_CANARY in r.payload.text for r in alice_looks] == [True]


async def test_list_document_chunks_requires_ownership(tenants: Tenants) -> None:
    t = tenants

    as_bob = await t.store.list_document_chunks(t.w1, t.alice_doc, t.bob)
    as_alice = await t.store.list_document_chunks(t.w1, t.alice_doc, t.alice)

    assert as_bob == []
    assert sorted(r.chunk_id for r in as_alice) == sorted(t.alice_chunks)


async def test_get_chunk_requires_ownership(tenants: Tenants) -> None:
    t = tenants
    chunk = t.alice_chunks[0]

    assert await t.store.get_chunk(t.w1, chunk, t.bob) is None
    own = await t.store.get_chunk(t.w1, chunk, t.alice)
    assert own is not None and own.payload.user_id == t.alice


async def test_delete_only_removes_the_callers_document(tenants: Tenants) -> None:
    t = tenants

    await t.store.delete_document_vectors(t.w1, t.alice_doc, t.bob)  # not Bob's
    assert len(await t.store.list_document_chunks(t.w1, t.alice_doc, t.alice)) == 2

    await t.store.delete_document_vectors(t.w1, t.alice_doc, t.alice)
    assert await t.store.list_document_chunks(t.w1, t.alice_doc, t.alice) == []
    assert len(await t.store.list_document_chunks(t.w1, t.bob_doc, t.bob)) == 2
