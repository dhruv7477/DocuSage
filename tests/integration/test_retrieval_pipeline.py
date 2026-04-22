"""
tests/integration/test_retrieval_pipeline.py
─────────────────────────────────────────────
Integration tests for the advanced retrieval pipeline.
All external dependencies mocked — no real Qdrant, LLM, or BM25 needed.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.user import User, UserRole
from src.providers.llm.base import LLMResponse
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.query_rewriter import QueryRewriter, RewriteStrategy
from src.retrieval.reranker import NoOpReranker
from src.vector_store.base import SearchResult, VectorPayload


def _make_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "alice@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _make_result(chunk_id: str, score: float, text: str = "chunk text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id="doc-1",
            chunk_index=0,
            text=text,
            source="doc.pdf",
            page_number=1,
        ),
    )


@pytest.fixture
def mock_embedder():
    e = AsyncMock()
    e.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return e


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.search = AsyncMock(return_value=[
        _make_result("c1", 0.9, "The refund policy is 30 days."),
        _make_result("c2", 0.8, "Items must be in original packaging."),
        _make_result("c3", 0.7, "Shipping costs are non-refundable."),
    ])
    return store


@pytest.fixture
def mock_rewriter():
    rewriter = AsyncMock(spec=QueryRewriter)
    rewriter.rewrite = AsyncMock(return_value=["What is the refund policy?"])
    return rewriter


class TestRetrievalPipelineBasic:

    @pytest.mark.asyncio
    async def test_returns_results_from_vector_search(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
            final_top_k=3,
        )
        user = _make_user()
        result = await pipeline.retrieve("refund policy", uuid.uuid4(), user)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_user_id_enforced_in_search_filter(self, mock_embedder, mock_vector_store):
        """user_id must always be in the SearchFilter — isolation guarantee."""
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        await pipeline.retrieve("query", uuid.uuid4(), user)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].user_id == str(user.id)

    @pytest.mark.asyncio
    async def test_workspace_id_enforced_in_search_filter(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        ws_id = uuid.uuid4()
        await pipeline.retrieve("query", ws_id, user)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].workspace_id == str(ws_id)

    @pytest.mark.asyncio
    async def test_document_id_filter_passed_through(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        doc_ids = ["doc-abc", "doc-def"]
        await pipeline.retrieve("query", uuid.uuid4(), user, document_ids=doc_ids)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].document_ids == doc_ids

    @pytest.mark.asyncio
    async def test_final_top_k_respected(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
            final_top_k=2,
        )
        user = _make_user()
        result = await pipeline.retrieve("query", uuid.uuid4(), user)
        assert len(result.results) <= 2


class TestRetrievalPipelineWithRewriter:

    @pytest.mark.asyncio
    async def test_rewriter_called_with_query(self, mock_embedder, mock_vector_store, mock_rewriter):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=mock_rewriter,
            reranker=NoOpReranker(),
            use_bm25=False,
            rewrite_strategy=RewriteStrategy.HYD,
        )
        user = _make_user()
        await pipeline.retrieve("what about refunds?", uuid.uuid4(), user)
        mock_rewriter.rewrite.assert_called_once()
        call_args = mock_rewriter.rewrite.call_args
        assert call_args.kwargs["query"] == "what about refunds?"

    @pytest.mark.asyncio
    async def test_rewritten_queries_stored_in_result(self, mock_embedder, mock_vector_store):
        rewriter = AsyncMock(spec=QueryRewriter)
        rewriter.rewrite = AsyncMock(return_value=[
            "What is the refund policy?",
            "How long do I have to return items?",
        ])
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=rewriter,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        result = await pipeline.retrieve("refund?", uuid.uuid4(), user)
        assert len(result.rewritten_queries) == 2

    @pytest.mark.asyncio
    async def test_multiple_queries_embed_separately(self, mock_vector_store):
        """Each rewritten query should produce a separate embed_query call."""
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

        rewriter = AsyncMock(spec=QueryRewriter)
        rewriter.rewrite = AsyncMock(return_value=["Query A", "Query B", "Query C"])

        pipeline = RetrievalPipeline(
            embedder=embedder,
            vector_store=mock_vector_store,
            rewriter=rewriter,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        await pipeline.retrieve("original query", uuid.uuid4(), user)
        assert embedder.embed_query.call_count == 3

    @pytest.mark.asyncio
    async def test_rewriter_failure_falls_back_to_original(self, mock_embedder, mock_vector_store):
        rewriter = AsyncMock(spec=QueryRewriter)
        rewriter.rewrite = AsyncMock(side_effect=Exception("rewriter failed"))

        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=rewriter,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        # Must not raise — should fall back to original query
        result = await pipeline.retrieve("original query", uuid.uuid4(), user)
        assert result.rewritten_queries == ["original query"]


class TestRetrievalPipelineWithReranker:

    @pytest.mark.asyncio
    async def test_reranker_called_with_original_query(self, mock_embedder, mock_vector_store):
        reranker = AsyncMock(spec=NoOpReranker)
        reranker.rerank = AsyncMock(return_value=[_make_result("c1", 0.9)])

        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=reranker,
            use_bm25=False,
        )
        user = _make_user()
        original = "What is the refund policy?"
        await pipeline.retrieve(original, uuid.uuid4(), user)

        reranker.rerank.assert_called_once()
        assert reranker.rerank.call_args.kwargs["query"] == original

    @pytest.mark.asyncio
    async def test_reranker_failure_falls_back_to_fused(self, mock_embedder, mock_vector_store):
        reranker = AsyncMock(spec=NoOpReranker)
        reranker.rerank = AsyncMock(side_effect=Exception("reranker exploded"))

        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=reranker,
            use_bm25=False,
            final_top_k=3,
        )
        user = _make_user()
        # Must not raise — returns unranked results
        result = await pipeline.retrieve("query", uuid.uuid4(), user)
        assert len(result.results) > 0


class TestRetrievalPipelineMetrics:

    @pytest.mark.asyncio
    async def test_result_tracks_vector_count(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
        )
        user = _make_user()
        result = await pipeline.retrieve("q", uuid.uuid4(), user)
        assert result.vector_result_count == 3  # 3 mock results

    @pytest.mark.asyncio
    async def test_result_tracks_final_count(self, mock_embedder, mock_vector_store):
        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rewriter=None,
            reranker=NoOpReranker(),
            use_bm25=False,
            final_top_k=2,
        )
        user = _make_user()
        result = await pipeline.retrieve("q", uuid.uuid4(), user)
        assert result.after_rerank_count == len(result.results)
