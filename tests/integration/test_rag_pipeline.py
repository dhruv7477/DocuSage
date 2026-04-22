"""
tests/integration/test_rag_pipeline.py
────────────────────────────────────────
End-to-end RAG pipeline tests.

Tests the complete: embed → search → prompt → generate → response chain.
Also verifies the critical user isolation guarantee:
    User A's query MUST NOT return User B's documents.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.user import User, UserRole
from src.providers.llm.base import LLMResponse
from src.services.prompt_builder import PromptBuilder
from src.services.rag_service import RAGService
from src.vector_store.base import SearchResult, VectorPayload


def _make_user(uid: uuid.UUID | None = None) -> User:
    user = User()
    user.id = uid or uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _make_result(text: str, user_id: str, score: float = 0.85) -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        score=score,
        payload=VectorPayload(
            user_id=user_id,
            workspace_id="ws-1",
            document_id=str(uuid.uuid4()),
            chunk_index=0,
            text=text,
            source="test.pdf",
            page_number=1,
        ),
    )


class TestRAGIsolationGuarantee:
    """
    Critical: verify that the user_id isolation filter is ALWAYS applied.
    These tests check that RAGService structurally enforces isolation
    regardless of what the caller passes.
    """

    @pytest.mark.asyncio
    async def test_search_filter_always_contains_user_id(self):
        """user_id in SearchFilter must match the authenticated user — always."""
        user_a = _make_user()
        user_b_id = str(uuid.uuid4())

        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024
        embedder.dimension = 1024

        store = AsyncMock()
        store.search.return_value = []

        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="Answer", model="mock", prompt_tokens=10, completion_tokens=5
        )

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        await rag.generate("query", uuid.uuid4(), user_a)

        call_kwargs = store.search.call_args.kwargs
        # The filter MUST use user_a's id, not any other user's id
        assert call_kwargs["search_filter"].user_id == str(user_a.id)
        assert call_kwargs["search_filter"].user_id != user_b_id

    @pytest.mark.asyncio
    async def test_two_users_same_workspace_get_different_results(self):
        """
        Simulate: User A and User B both in Workspace W.
        Each user's search must be filtered to their own documents.
        """
        user_a = _make_user()
        user_b = _make_user()
        ws_id = uuid.uuid4()

        # User A's docs
        user_a_result = _make_result("User A's private document content.", str(user_a.id))
        # User B's docs
        user_b_result = _make_result("User B's private document content.", str(user_b.id))

        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024
        embedder.dimension = 1024

        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="Answer.", model="mock", prompt_tokens=10, completion_tokens=5
        )

        # Store returns different results based on user_id filter
        async def filtered_search(workspace_id, query_vector, search_filter, **kwargs):
            if search_filter.user_id == str(user_a.id):
                return [user_a_result]
            elif search_filter.user_id == str(user_b.id):
                return [user_b_result]
            return []

        store = AsyncMock()
        store.search = filtered_search

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)

        # User A queries — should get User A's docs passed to LLM
        await rag.generate("query", ws_id, user_a)
        prompt_a = llm.generate.call_args[0][0]  # messages list
        system_a = prompt_a[0].content
        assert "User A's private document content" in system_a
        assert "User B's private document content" not in system_a

        # User B queries — should get User B's docs passed to LLM
        await rag.generate("query", ws_id, user_b)
        prompt_b = llm.generate.call_args[0][0]
        system_b = prompt_b[0].content
        assert "User B's private document content" in system_b
        assert "User A's private document content" not in system_b

    @pytest.mark.asyncio
    async def test_workspace_id_always_in_filter(self):
        """workspace_id isolation is also enforced — not just user_id."""
        user = _make_user()
        ws_id = uuid.uuid4()

        store = AsyncMock()
        store.search.return_value = []
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024
        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="A", model="m", prompt_tokens=1, completion_tokens=1
        )

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        await rag.generate("q", ws_id, user)

        call_kwargs = store.search.call_args.kwargs
        assert call_kwargs["search_filter"].workspace_id == str(ws_id)


class TestRAGFullPipeline:
    """Tests the complete pipeline from query to answer."""

    @pytest.mark.asyncio
    async def test_full_pipeline_embed_search_generate(self):
        """Verify embed → search → prompt → generate all called in order."""
        user = _make_user()
        ws_id = uuid.uuid4()
        query = "What is the refund policy?"
        context_text = "Refunds are allowed within 30 days."

        embedder = AsyncMock()
        query_vector = [0.42] * 1024
        embedder.embed_query.return_value = query_vector
        embedder.dimension = 1024

        store = AsyncMock()
        store.search.return_value = [
            _make_result(context_text, str(user.id), score=0.91)
        ]

        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="Refunds are allowed within 30 days of purchase.",
            model="llama3.3:70b",
            prompt_tokens=120,
            completion_tokens=15,
        )

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        response = await rag.generate(query, ws_id, user)

        # 1. Embedder was called with the query
        embedder.embed_query.assert_called_once_with(query)

        # 2. Vector store was searched with the embedded query
        search_call = store.search.call_args
        assert search_call.kwargs["query_vector"] == query_vector

        # 3. LLM was called with a message list
        llm.generate.assert_called_once()
        messages = llm.generate.call_args[0][0]
        assert len(messages) >= 2  # At least system + user

        # 4. Context appears in system message
        system_msg = messages[0].content
        assert context_text in system_msg

        # 5. Final answer matches LLM output
        assert "30 days" in response.answer

    @pytest.mark.asyncio
    async def test_history_injected_into_prompt(self):
        """Prior conversation messages should appear in the built prompt."""
        from src.providers.llm.base import LLMMessage

        user = _make_user()
        history = [
            LLMMessage(role="user", content="What is your name?"),
            LLMMessage(role="assistant", content="I am a knowledge assistant."),
        ]

        store = AsyncMock()
        store.search.return_value = []
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024
        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="Answer", model="mock", prompt_tokens=50, completion_tokens=5
        )

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        await rag.generate("Follow-up question", uuid.uuid4(), user, history_messages=history)

        messages = llm.generate.call_args[0][0]
        contents = [m.content for m in messages]

        assert "What is your name?" in contents
        assert "I am a knowledge assistant." in contents

    @pytest.mark.asyncio
    async def test_sources_cite_correct_documents(self):
        """Sources in response should reference the retrieved documents."""
        user = _make_user()
        doc_id = str(uuid.uuid4())
        result = SearchResult(
            chunk_id=str(uuid.uuid4()),
            score=0.93,
            payload=VectorPayload(
                user_id=str(user.id),
                workspace_id="ws-1",
                document_id=doc_id,
                chunk_index=2,
                text="Policy content here.",
                source="company_policy.pdf",
                page_number=4,
            ),
        )

        store = AsyncMock()
        store.search.return_value = [result]
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024
        llm = AsyncMock()
        llm.generate.return_value = LLMResponse(
            content="Answer", model="mock", prompt_tokens=20, completion_tokens=5
        )

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        response = await rag.generate("Query", uuid.uuid4(), user)

        assert len(response.sources) == 1
        source = response.sources[0]
        assert source.document_id == doc_id
        assert source.source_filename == "company_policy.pdf"
        assert source.page_number == 4
        assert source.relevance_score == 0.93

    @pytest.mark.asyncio
    async def test_streaming_yields_all_tokens(self):
        """Stream must yield every token from the LLM in order."""
        user = _make_user()
        expected_tokens = ["The ", "answer ", "is ", "42."]

        store = AsyncMock()
        store.search.return_value = []
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024

        async def fake_stream(messages, config=None):
            for t in expected_tokens:
                yield t

        llm = AsyncMock()
        llm.stream = fake_stream

        rag = RAGService(embedder=embedder, vector_store=store, llm=llm)
        collected = []
        async for token in rag.stream("What is the answer?", uuid.uuid4(), user):
            collected.append(token)

        assert collected == expected_tokens
        assert "".join(collected) == "The answer is 42."
