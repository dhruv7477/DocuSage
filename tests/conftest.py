"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the entire test suite.

Fixture hierarchy:
    settings  →  app  →  client       (integration tests)
    mock_db   →  service               (unit tests)
    mock_embedder / mock_llm / mock_vs  (RAG unit tests)

All fixtures are async-safe (pytest-asyncio auto mode).
No real DB, no real LLM, no real MinIO are needed to run the test suite.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Environment Setup (must happen before any app imports) ────────
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("APP_DEBUG", "true")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-32chars")
os.environ.setdefault("POSTGRES_PASSWORD", "testpassword")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "privateai_test")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.3:70b")
os.environ.setdefault("EMBEDDING_PROVIDER", "bge")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-m3")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("VECTOR_STORE", "qdrant")
os.environ.setdefault("STORAGE_BACKEND", "minio")


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def workspace_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
def chat_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000003")


@pytest.fixture
def mock_user(user_id):
    from src.models.user import User, UserRole
    user = User()
    user.id = user_id
    user.email = "alice@example.com"
    user.full_name = "Alice"
    user.role = UserRole.MEMBER
    user.is_active = True
    user.is_email_verified = True
    return user


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    return db


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_query.return_value = [0.01] * 1024
    embedder.embed_documents.return_value = [[0.01] * 1024]
    embedder.health.return_value = True
    embedder.dimension = 1024
    embedder.provider_name = "mock-embedder"
    return embedder


@pytest.fixture
def mock_llm():
    from src.providers.llm.base import LLMResponse
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(
        content="This is a mock answer from the LLM.",
        model="mock-llm",
        prompt_tokens=50,
        completion_tokens=15,
    )
    llm.health.return_value = True
    llm.provider_name = "mock-llm"

    async def _fake_stream(messages, config=None):
        for token in ["This ", "is ", "a ", "mock ", "answer."]:
            yield token

    llm.stream = _fake_stream
    return llm


@pytest.fixture
def mock_vector_store():
    from src.vector_store.base import SearchResult, VectorPayload
    store = AsyncMock()
    store.health.return_value = True
    store.collection_exists.return_value = True
    store.search.return_value = [
        SearchResult(
            chunk_id=str(uuid.uuid4()),
            score=0.87,
            payload=VectorPayload(
                user_id="00000000-0000-0000-0000-000000000001",
                workspace_id="00000000-0000-0000-0000-000000000002",
                document_id=str(uuid.uuid4()),
                chunk_index=0,
                text="The return policy allows refunds within 30 days of purchase.",
                source="policy.pdf",
                page_number=2,
                chunk_total=10,
            ),
        )
    ]
    return store


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload.return_value = "workspaces/ws-1/documents/user-1/doc-1/file.pdf"
    storage.download.return_value = b"fake file content"
    storage.delete.return_value = None
    return storage


@pytest.fixture
def rag_service(mock_embedder, mock_vector_store, mock_llm):
    from src.services.rag_service import RAGService
    return RAGService(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm=mock_llm,
        top_k=5,
        score_threshold=0.5,
    )


@pytest.fixture
def auth_headers(mock_user):
    from src.core.security import create_access_token
    token = create_access_token(subject=str(mock_user.id), token_type="access")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return b"This is a sample text document for testing. " * 20


@pytest.fixture
def sample_html_bytes() -> bytes:
    return (
        b"<html><body>"
        b"<h1>Test Document</h1>"
        b"<p>This is the main content of the test document.</p>"
        b"<p>It contains multiple paragraphs for testing purposes.</p>"
        b"</body></html>"
    )


# ── Phase 4: Advanced Retrieval Fixtures ─────────────────────────

@pytest.fixture
def mock_query_understanding():
    """
    Mock `QueryUnderstanding` that returns a 1-query enrichment with
    no rewriting and no HyDE. Tests that need to exercise rewriting
    instantiate the real class with a mock LLM.
    """
    from src.retrieval.query_understanding import QueryEnrichment

    qu = AsyncMock()
    async def _enrich(query: str, history: list[tuple[str, str]] | None = None):
        return QueryEnrichment(
            original=query,
            standalone=query,
            queries=[query],
            rewritten=False,
            used_hyde=False,
        )
    qu.enrich_query = AsyncMock(side_effect=_enrich)
    return qu


@pytest.fixture
def mock_reranker():
    """Mock reranker that returns results in the same order (NoOp)."""
    from src.retrieval.reranker import NoOpReranker
    return NoOpReranker()


@pytest.fixture
def hybrid_retriever(mock_embedder, mock_vector_store):
    """Real HybridRetriever wired around the mocked vector store."""
    from src.retrieval.hybrid import HybridRetriever
    return HybridRetriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        candidate_k=10,
    )


@pytest.fixture
def rag_service_advanced(
    mock_embedder,
    mock_vector_store,
    mock_llm,
    hybrid_retriever,
    mock_reranker,
    mock_query_understanding,
):
    """RAGService wired with the Phase 3 retrieval stack."""
    from src.services.rag_service import RAGService
    return RAGService(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm=mock_llm,
        retriever=hybrid_retriever,
        reranker=mock_reranker,
        query_understanding=mock_query_understanding,
        top_k=5,
        abstention_threshold=0.0,  # tests don't exercise abstention by default
    )


@pytest.fixture
def eval_service(mock_llm, mock_embedder):
    """EvalService with mocked LLM and embedder."""
    from src.services.eval_service import EvalService
    return EvalService(llm=mock_llm, embedder=mock_embedder)