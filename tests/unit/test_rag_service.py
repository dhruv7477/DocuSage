"""
tests/unit/test_rag_service.py
────────────────────────────────
Unit tests for the Phase 3 `RAGService` — all collaborators mocked.
Tests the full pipeline: query-understand → hybrid retrieve → rerank →
prompt build → LLM call.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.core.exceptions import LLMProviderError
from src.configs.config import Settings
from src.models.user import User, UserRole
from src.providers.llm.base import EMPTY_ANSWER_MESSAGE, LLMResponse
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_understanding import QueryEnrichment
from src.retrieval.reranker import BaseReranker, NoOpReranker
from src.services.prompt_builder import PromptBuilder
from src.services.rag_service import RAGService, _extract_sources
from src.vector_store.base import SearchResult, VectorPayload


def _make_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _make_search_result(
    text: str, score: float = 0.85, doc_id: str = "doc-1"
) -> SearchResult:
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
        # Never decline: these tests don't exercise the answer gate.
        reranker_threshold=0.0,
        cosine_threshold=0.0,
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
    async def test_generate_search_includes_user_id(
        self, rag_service, mock_vector_store
    ):
        user = _make_user()
        ws_id = uuid.uuid4()
        await rag_service.generate("Query", ws_id, user)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["search_filter"].user_id == str(user.id)

    @pytest.mark.asyncio
    async def test_generate_search_includes_workspace_id(
        self, rag_service, mock_vector_store
    ):
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
    async def test_document_id_filter_passed_to_search(
        self, rag_service, mock_vector_store
    ):
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


class TestPrepareStream:
    @pytest.mark.asyncio
    async def test_on_stage_callback_sequence(self, rag_service, mock_llm):
        async def _fake_stream(messages, config):
            yield "token"

        mock_llm.stream = _fake_stream
        stages: list[str] = []
        outcome, token_iter = await rag_service.prepare_stream(
            "Query",
            uuid.uuid4(),
            _make_user(),
            on_stage=stages.append,
        )
        [t async for t in token_iter]
        assert stages == ["retrieving", "reranking"]

    @pytest.mark.asyncio
    async def test_outcome_carries_built_prompt(self, rag_service, mock_llm):
        async def _fake_stream(messages, config):
            yield "token"

        mock_llm.stream = _fake_stream
        outcome, _ = await rag_service.prepare_stream(
            "Query",
            uuid.uuid4(),
            _make_user(),
        )
        assert outcome.prompt is not None
        assert outcome.prompt.included_results  # the mocked chunk made it in


class TestExtractSources:
    def test_deduplicates_by_document_and_page(self):
        results = [
            _make_search_result("Chunk A", score=0.9, doc_id="doc-1"),
            _make_search_result(
                "Chunk B", score=0.7, doc_id="doc-1"
            ),  # Same doc, same page
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


class TestEmptyAnswer:
    """A reasoning model can return no text at all; the user gets a retry message."""

    @pytest.mark.asyncio
    async def test_empty_streamed_answer_raises_the_retry_message(
        self, rag_service, mock_llm
    ):
        async def _nothing(messages, config):
            return
            yield  # an async generator that yields nothing

        mock_llm.stream = _nothing
        _, tokens = await rag_service.prepare_stream(
            "Query", uuid.uuid4(), _make_user()
        )
        with pytest.raises(LLMProviderError) as caught:
            _ = [token async for token in tokens]
        assert caught.value.message == EMPTY_ANSWER_MESSAGE

    @pytest.mark.asyncio
    async def test_empty_generated_answer_raises_the_retry_message(
        self, rag_service, mock_llm
    ):
        mock_llm.generate.return_value = LLMResponse(content="  \n", model="m")
        with pytest.raises(LLMProviderError) as caught:
            await rag_service.generate("Query", uuid.uuid4(), _make_user())
        assert caught.value.message == EMPTY_ANSWER_MESSAGE


class _FixedReranker(BaseReranker):
    """Gives the candidates the reranker scores a test needs, in order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def rerank(self, query, results, top_k):
        return [
            SearchResult(chunk_id=r.chunk_id, score=s, payload=r.payload)
            for r, s in zip(results, self._scores)
        ][:top_k]


def _gated_service(
    embedder, store, llm, query_understanding, *, reranker_top, cosine_top
):
    store.search.return_value = (
        [_make_search_result("A chunk.", score=cosine_top)]
        if cosine_top is not None
        else []
    )
    return RAGService(
        embedder=embedder,
        vector_store=store,
        llm=llm,
        # A full candidate list (1 of 1): a workspace too big to be small.
        retriever=HybridRetriever(embedder=embedder, vector_store=store, candidate_k=1),
        reranker=_FixedReranker([reranker_top]),
        query_understanding=query_understanding,
        reranker_threshold=0.5,
        cosine_threshold=0.48,
    )


class TestAnswerGate:
    """
    The cross-encoder ranks well but misses answers worded unlike the
    question ("the candidate" against a resume), so a question is declined
    without an LLM call only when BOTH its top reranker score and its top
    dense cosine are low. Everything else goes to the LLM, which can still
    say the documents don't cover it.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reranker_top", "cosine_top", "declined"),
        [
            (0.0, 0.55, False),  # the resume case: reranker misses, cosine doesn't
            (0.0, 0.45, True),  # clearly irrelevant on both
            (0.9, 0.45, False),  # a strong reranker hit vetoes a low cosine
            (0.5, 0.40, False),  # at the reranker cut-off: not below it
            (0.1, 0.48, False),  # at the cosine cut-off: not below it
            (0.0, 0.4979, False),  # a one-page resume asked for the name
        ],
    )
    async def test_declines_only_when_both_scores_are_low(
        self,
        mock_embedder,
        mock_vector_store,
        mock_llm,
        mock_query_understanding,
        reranker_top,
        cosine_top,
        declined,
    ):
        svc = _gated_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            reranker_top=reranker_top,
            cosine_top=cosine_top,
        )
        outcome, _ = await svc.prepare_stream("Question", uuid.uuid4(), _make_user())
        assert svc._is_abstaining(outcome) is declined

    @pytest.mark.asyncio
    async def test_no_candidates_declines(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        svc = _gated_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            reranker_top=0.9,
            cosine_top=None,
        )
        outcome, _ = await svc.prepare_stream("Question", uuid.uuid4(), _make_user())
        assert svc._is_abstaining(outcome)

    @pytest.mark.asyncio
    async def test_each_decision_is_logged_once_without_the_question(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        svc = _gated_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            reranker_top=0.0,
            cosine_top=0.55,
        )
        with patch("src.services.rag_service.log") as log:
            await svc.prepare_stream(
                "What is the secret question?", uuid.uuid4(), _make_user()
            )
        gate = [c for c in log.info.call_args_list if c.args[0] == "rag_gate"]
        assert len(gate) == 1
        assert gate[0].kwargs == {
            "zone": "llm",
            "reranker_top": 0.0,
            "cosine_top": 0.55,
            "candidates": 1,
            "small_workspace": False,
            "reranker_threshold": 0.5,
            "cosine_threshold": 0.48,
        }
        assert "secret" not in str(gate[0])

    def test_defaults_are_the_tuned_cut_offs(self):
        settings = Settings()
        assert settings.abstention_reranker_threshold == 0.5
        assert settings.abstention_cosine_threshold == 0.48


def _chunk(
    doc_id: str, index: int, text: str = "", cosine: float = 0.5
) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc_id}#{index}",
        score=cosine,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id=doc_id,
            chunk_index=index,
            text=text or f"Passage {index} of {doc_id}.",
            source=f"{doc_id}.pdf",
            page_number=index + 1,
        ),
    )


class _ScoredReranker(BaseReranker):
    """Scores each candidate by chunk id (0.0 when unlisted), best first."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def rerank(self, query, results, top_k):
        scored = [
            SearchResult(
                chunk_id=r.chunk_id,
                score=self._scores.get(r.chunk_id, 0.0),
                payload=r.payload,
            )
            for r in results
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]


def _workspace_service(
    embedder,
    store,
    llm,
    query_understanding,
    chunks,
    *,
    scores=None,
    candidate_k=20,
    context_tokens=2500,
):
    store.search.return_value = chunks
    return RAGService(
        embedder=embedder,
        vector_store=store,
        llm=llm,
        prompt_builder=PromptBuilder(max_context_tokens=context_tokens),
        retriever=HybridRetriever(
            embedder=embedder, vector_store=store, candidate_k=candidate_k
        ),
        reranker=_ScoredReranker(scores or {}),
        query_understanding=query_understanding,
        top_k=3,
        reranker_threshold=0.5,
        cosine_threshold=0.48,
    )


class TestSmallWorkspace:
    """
    When retrieval returns every chunk the user can search and they all fit
    the prompt, each document's first chunk joins the context (a resume's
    name is in its first chunk, and the reranker scores "the name of the
    candidate" near zero everywhere), and the gate leaves the decision to
    the LLM.
    """

    @staticmethod
    def _two_documents(cosine: float = 0.5) -> list[SearchResult]:
        return [
            _chunk(d, i, cosine=cosine) for d in ("doc-a", "doc-b") for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_each_documents_first_chunk_joins_the_context(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        # The first chunks rank last on the reranker, outside the top 3.
        scores = {"doc-a#3": 0.9, "doc-b#2": 0.8, "doc-a#1": 0.7, "doc-b#4": 0.6}
        svc = _workspace_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            self._two_documents(),
            scores=scores,
        )
        outcome, _ = await svc.prepare_stream("Name?", uuid.uuid4(), _make_user())
        included = [r.chunk_id for r in outcome.prompt.included_results]
        assert included == ["doc-a#3", "doc-b#2", "doc-a#1", "doc-a#0", "doc-b#0"]
        assert outcome.small_workspace is True

    @pytest.mark.asyncio
    async def test_a_first_chunk_already_in_the_top_is_not_repeated(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        chunks = [_chunk("doc-a", i) for i in range(5)]
        svc = _workspace_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            chunks,
            scores={"doc-a#0": 0.9, "doc-a#1": 0.8, "doc-a#2": 0.7},
        )
        outcome, _ = await svc.prepare_stream("Name?", uuid.uuid4(), _make_user())
        assert [r.chunk_id for r in outcome.results] == [
            "doc-a#0",
            "doc-a#1",
            "doc-a#2",
        ]

    @pytest.mark.asyncio
    async def test_low_scores_go_to_the_llm(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        svc = _workspace_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            self._two_documents(cosine=0.45),
        )
        with patch("src.services.rag_service.log") as log:
            outcome, _ = await svc.prepare_stream("Name?", uuid.uuid4(), _make_user())
        assert not svc._is_abstaining(outcome)
        [gate] = [c for c in log.info.call_args_list if c.args[0] == "rag_gate"]
        assert (gate.kwargs["zone"], gate.kwargs["small_workspace"]) == ("llm", True)

    @pytest.mark.asyncio
    async def test_a_full_candidate_list_is_not_small(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        # Ten hits for ten asked: there may be more chunks than were returned.
        svc = _workspace_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            self._two_documents(cosine=0.45),
            candidate_k=10,
        )
        outcome, _ = await svc.prepare_stream("Name?", uuid.uuid4(), _make_user())
        assert outcome.small_workspace is False
        assert len(outcome.results) == 3
        assert svc._is_abstaining(outcome)

    @pytest.mark.asyncio
    async def test_chunks_that_do_not_fit_the_prompt_are_not_small(
        self, mock_embedder, mock_vector_store, mock_llm, mock_query_understanding
    ):
        long_text = "word " * 120
        chunks = [
            _chunk("doc-a", i, text=f"{i} {long_text}", cosine=0.45) for i in range(5)
        ]
        svc = _workspace_service(
            mock_embedder,
            mock_vector_store,
            mock_llm,
            mock_query_understanding,
            chunks,
            context_tokens=300,
        )
        outcome, _ = await svc.prepare_stream("Name?", uuid.uuid4(), _make_user())
        assert outcome.small_workspace is False
        assert svc._is_abstaining(outcome)
