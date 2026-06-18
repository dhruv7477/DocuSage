"""
tests/unit/test_doc_summary_service.py
───────────────────────────────────────
Map-reduce summarizer tests — vector store and LLM mocked.
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from src.models.user import User, UserRole
from src.providers.llm.base import LLMResponse
from src.services.doc_summary_service import DocSummaryService, _pack_segments
from src.vector_store.base import SearchResult, VectorPayload

pytestmark = pytest.mark.unit


def _make_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _chunk(text: str, index: int, doc_id: str = "doc-1") -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        score=1.0,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id=doc_id,
            chunk_index=index,
            text=text,
            source="report.pdf",
            page_number=index + 1,
        ),
    )


def _mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(content="- bullet", model="test")

    async def _stream(messages, config=None):
        for tok in ["Executive ", "summary."]:
            yield tok

    llm.stream = _stream
    return llm


def _service(store, llm, **overrides) -> DocSummaryService:
    defaults = dict(
        map_token_budget=2500,
        section_max_tokens=350,
        final_max_tokens=700,
        max_total_chunks=500,
    )
    defaults.update(overrides)
    return DocSummaryService(store, llm, **defaults)


class TestSinglePass:

    @pytest.mark.asyncio
    async def test_small_doc_skips_map_phase(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk("Short chunk one.", 0),
            _chunk("Short chunk two.", 1),
        ]
        llm = _mock_llm()
        outcome, stream = await _service(store, llm).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        tokens = [t async for t in stream]

        assert outcome.map_calls == 0
        assert outcome.total_chunks == 2
        assert not outcome.truncated
        assert "".join(tokens) == "Executive summary."
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_document_streams_graceful_note(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = []
        outcome, stream = await _service(store, _mock_llm()).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        text = "".join([t async for t in stream])
        assert "couldn't find any indexed content" in text
        assert outcome.total_chunks == 0

    @pytest.mark.asyncio
    async def test_sources_one_per_document(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [_chunk("text", 0)]
        outcome, _ = await _service(store, _mock_llm()).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "a.pdf"), ("doc-2", "b.pdf")],
            query="summarize",
        )
        assert [s.document_id for s in outcome.sources] == ["doc-1", "doc-2"]


class TestMapReduce:

    @pytest.mark.asyncio
    async def test_large_doc_triggers_map_calls(self):
        # ~400 tokens per chunk × 20 chunks ≫ 1000-token budget
        big_text = "word " * 400
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk(big_text, i) for i in range(20)
        ]
        llm = _mock_llm()
        outcome, stream = await _service(
            store, llm, map_token_budget=1000
        ).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        [t async for t in stream]

        assert outcome.map_calls > 0
        assert llm.generate.call_count == outcome.map_calls

    @pytest.mark.asyncio
    async def test_chunk_truncation_guard(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk("text", i) for i in range(10)
        ]
        outcome, _ = await _service(
            store, _mock_llm(), max_total_chunks=5
        ).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        assert outcome.truncated
        assert outcome.total_chunks == 5


class TestPackSegments:

    def test_segments_respect_budget(self):
        paragraphs = "\n\n".join(["word " * 100] * 10)  # ~100 tokens each
        segments = _pack_segments([paragraphs], budget=250)
        assert len(segments) > 1
        # No segment may grossly exceed the budget (a single paragraph
        # over budget is allowed — chunks are never split).
        from src.services.prompt_builder import count_tokens
        for seg in segments:
            assert count_tokens(seg) <= 250 + 110

    def test_documents_never_mix_within_segment(self):
        doc_a = "AAA " * 50
        doc_b = "BBB " * 50
        segments = _pack_segments([doc_a, doc_b], budget=10_000)
        # Even with a huge budget, separate input texts yield separate segments.
        assert len(segments) == 2
        assert "BBB" not in segments[0]
        assert "AAA" not in segments[1]
