"""
tests/unit/test_rag_service.py
────────────────────────────────
Unit tests for the Phase 3 `RAGService` — all collaborators mocked.
Tests the full pipeline: query-understand → hybrid retrieve → rerank →
prompt build → LLM call.
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from src.models.user import User, UserRole
from src.providers.llm.base import LLMResponse
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_understanding import QueryEnrichment
from src.retrieval.reranker import NoOpReranker
from src.services.rag_service import RAGService, _extract_sources
from src.vector_store.base import SearchResult, VectorPayload


def _make_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _make_search_result(text: str, score: float = 0.85, doc_id: str = "doc-1") -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        score=score,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id=doc_id,
            chunk_index=0,
            text=text,
            source="report.pdf",
            page_number=1,
        ),
    )


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_query.return_value = [0.1] * 1024
    embedder.embed_documents.return_value = [[0.1] * 1024]
    embedder.dimension = 1024
    return embedder


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.search.return_value = [
        _make_search_result("The refund policy allows returns within 30 days.")
    ]
    # Sparse branch is non-essential for these tests; empty list is fine.
    store.text_search.return_value = []
    return store


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(
        content="Based on the documents, the refund policy is 30 days.",
        model="qwen2.5:7b-instruct",
        prompt_tokens=100,
        completion_tokens=20,
    )
    return llm


@pytest.fixture
def mock_query_understanding():
    qu = AsyncMock()

    async def _enrich(query, history=None):
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
def rag_service(mock_embedder, mock_vector_store, mock_llm, mock_query_understanding):
    retriever = HybridRetriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        candidate_k=5,
    )
    return RAGService(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm=mock_llm,
        retriever=retriever,
        reranker=NoOpReranker(),
        query_understanding=mock_query_understanding,
        top_k=5,
        abstention_threshold=0.0,  # tests don't exercise abstention
    )


class TestRAGServiceGenerate:

    @pytest.mark.asyncio
    async def test_generate_returns_answer(self, rag_service):
        user = _make_user()
        ws_id = uuid.uuid4()
        response = await rag_service.generate("What is the refund policy?", ws_id, user)
        assert "30 days" in response.answer

    @pytest.mark.asyncio
    async def test_generate_calls_embed_query(self, rag_service, mock_embedder):
        user = _make_user()
        await rag_service.generate("Test query", uuid.uuid4(), user)
        mock_embedder.embed_query.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_generate_search_includes_user_id(self, rag_service, mock_vector_store):
        user = _make_user()
        ws_id = uuid.uuid4()
        await rag_service.generate("Query", ws_id, user)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].user_id == str(user.id)

    @pytest.mark.asyncio
    async def test_generate_search_includes_workspace_id(self, rag_service, mock_vector_store):
        user = _make_user()
        ws_id = uuid.uuid4()
        await rag_service.generate("Query", ws_id, user)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].workspace_id == str(ws_id)

    @pytest.mark.asyncio
    async def test_sources_returned(self, rag_service):
        user = _make_user()
        response = await rag_service.generate("Query", uuid.uuid4(), user)
        assert len(response.sources) > 0
        assert response.sources[0].source_filename == "report.pdf"

    @pytest.mark.asyncio
    async def test_no_results_still_generates(self, rag_service, mock_vector_store):
        """Empty retrieval no longer raises — the LLM is called with the abstention prompt."""
        mock_vector_store.search.return_value = []
        user = _make_user()
        response = await rag_service.generate("Unknown query", uuid.uuid4(), user)
        assert response.answer  # LLM still called
        assert response.abstained is True

    @pytest.mark.asyncio
    async def test_document_id_filter_passed_to_search(self, rag_service, mock_vector_store):
        user = _make_user()
        doc_ids = ["doc-abc", "doc-def"]
        await rag_service.generate("Query", uuid.uuid4(), user, document_ids=doc_ids)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].document_ids == doc_ids


class TestRAGServiceStream:

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, rag_service, mock_llm):
        async def _fake_stream(messages, config):
            for token in ["Hello", " world", "."]:
                yield token

        mock_llm.stream = _fake_stream
        user = _make_user()
        tokens = []
        async for token in rag_service.stream("Query", uuid.uuid4(), user):
            tokens.append(token)
        assert tokens == ["Hello", " world", "."]


class TestExtractSources:

    def test_deduplicates_by_document_and_page(self):
        results = [
            _make_search_result("Chunk A", score=0.9, doc_id="doc-1"),
            _make_search_result("Chunk B", score=0.7, doc_id="doc-1"),  # Same doc, same page
        ]
        sources = _extract_sources(results)
        assert len(sources) == 1
        assert sources[0].relevance_score == 0.9  # Keeps highest score

    def test_different_docs_not_deduplicated(self):
        results = [
            _make_search_result("Chunk A", score=0.9, doc_id="doc-1"),
            _make_search_result("Chunk B", score=0.8, doc_id="doc-2"),
        ]
        sources = _extract_sources(results)
        assert len(sources) == 2

    def test_sources_sorted_by_score_descending(self):
        results = [
            _make_search_result("Low", score=0.6, doc_id="doc-1"),
            _make_search_result("High", score=0.95, doc_id="doc-2"),
            _make_search_result("Mid", score=0.75, doc_id="doc-3"),
        ]
        sources = _extract_sources(results)
        scores = [s.relevance_score for s in sources]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results_returns_empty(self):
        assert _extract_sources([]) == []
