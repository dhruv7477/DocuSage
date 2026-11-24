"""
tests/unit/test_doc_summary_service.py
───────────────────────────────────────
Map-reduce summarizer tests — vector store and LLM mocked.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.configs.config import Settings
from src.core.exceptions import LLMProviderError
from src.models.user import User, UserRole
from src.providers.llm.base import EMPTY_ANSWER_MESSAGE, LLMResponse
from src.services.doc_summary_service import (
    _FINAL_SYSTEM_PROMPT,
    _MAP_SYSTEM_PROMPT,
    OVERVIEW_NOTE,
    DocSummaryService,
    _pack_segments,
)
from src.services.prompt_builder import count_tokens
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

    # Both the map phase and the final pass stream(); stream() is a sync
    # call that returns an async iterator. MagicMock records the call args
    # (so tests can assert count / timeout) while side_effect yields tokens.
    async def _stream_impl(messages, config=None, *, timeout=None):
        for tok in ["Executive ", "summary."]:
            yield tok

    llm.stream = MagicMock(side_effect=_stream_impl)
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


def _headed(text: str, index: int, doc_id: str, heading: str) -> SearchResult:
    chunk = _chunk(text, index, doc_id)
    chunk.payload.heading_path = [heading]
    return chunk


def _long_doc(doc_id: str, word: str, n: int = 12) -> list[SearchResult]:
    return [
        _headed(f"{word} section {i}. " * 40, i, doc_id, f"{word} heading {i}")
        for i in range(n)
    ]


def _overview_material(llm) -> str:
    """The excerpts sent with the single overview call."""
    messages = llm.stream.call_args.args[0]
    return messages[-1].content.split("MATERIAL:\n", 1)[1]


class TestOverview:
    """Summaries off: one call over each document's headings and opening chunks."""

    async def _run(self, docs: dict[str, list[SearchResult]], budget: int = 1000):
        store = AsyncMock()
        store.list_document_chunks.side_effect = lambda ws, doc_id, user: docs[doc_id]
        llm = _mock_llm()
        outcome, stream = await _service(
            store, llm, overview_token_budget=budget
        ).prepare_overview_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[(doc_id, f"{doc_id}.pdf") for doc_id in docs],
            query="Summary of the document",
        )
        text = "".join([t async for t in stream])
        return store, llm, outcome, text

    @pytest.mark.asyncio
    async def test_one_call_never_a_search(self):
        store, llm, _, _ = await self._run({"doc-1": _long_doc("doc-1", "Alpha")})
        assert llm.stream.call_count == 1
        llm.generate.assert_not_called()
        store.search.assert_not_called()
        store.text_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_headings_and_opening_chunks_within_the_budget(self):
        docs = {
            "doc-1": _long_doc("doc-1", "Alpha"),
            "doc-2": _long_doc("doc-2", "Beta"),
        }
        _, llm, outcome, _ = await self._run(docs, budget=1000)
        material = _overview_material(llm)

        assert count_tokens(material) <= 1000
        for word in ("Alpha", "Beta"):
            assert f"{word} heading 0" in material
            assert f"{word} heading 11" in material  # the outline covers the document
            assert f"{word} section 0." in material  # opening chunk
            assert f"{word} section 11." not in material  # never the end
        assert outcome.truncated
        assert [s.document_id for s in outcome.sources] == ["doc-1", "doc-2"]

    @pytest.mark.asyncio
    async def test_a_short_document_is_sent_whole(self):
        doc = [_headed("Short resume text.", 0, "doc-1", "Jordan Lee")]
        _, llm, outcome, _ = await self._run({"doc-1": doc})
        material = _overview_material(llm)
        assert "Jordan Lee" in material
        assert "Short resume text." in material
        assert not outcome.truncated

    @pytest.mark.asyncio
    async def test_starts_with_the_note(self):
        _, _, _, text = await self._run({"doc-1": _long_doc("doc-1", "Alpha")})
        assert text == OVERVIEW_NOTE + "Executive summary."
        assert OVERVIEW_NOTE.startswith(
            "Short overview: full summaries are turned off in this demo, so this "
            "covers each document's headings and opening sections."
        )
        assert chr(0x2014) not in OVERVIEW_NOTE  # no em dash

    @pytest.mark.asyncio
    async def test_many_documents_share_the_budget(self):
        docs = {f"doc-{i}": _long_doc(f"doc-{i}", f"Word{i}") for i in range(5)}
        _, llm, _, _ = await self._run(docs, budget=1500)
        material = _overview_material(llm)
        assert count_tokens(material) <= 1500
        for i in range(5):
            assert f"Word{i} section 0." in material

    @pytest.mark.asyncio
    async def test_no_indexed_content_streams_a_note_without_a_call(self):
        _, llm, outcome, text = await self._run({"doc-1": []})
        assert "couldn't find any indexed content" in text
        llm.stream.assert_not_called()
        assert outcome.total_chunks == 0


class TestEmptyAnswer:
    """gpt-oss can spend its whole budget on reasoning and return no text."""

    @staticmethod
    def _silent_llm() -> AsyncMock:
        async def _nothing(messages, config=None, *, timeout=None):
            return
            yield  # an async generator that yields nothing

        llm = _mock_llm()
        llm.stream = MagicMock(side_effect=_nothing)
        return llm

    @pytest.mark.asyncio
    async def test_empty_overview_raises_the_retry_message_without_the_note(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [_chunk("Some text.", 0)]
        _, stream = await _service(store, self._silent_llm()).prepare_overview_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="Give a summary",
        )
        received: list[str] = []
        with pytest.raises(LLMProviderError) as caught:
            async for token in stream:
                received.append(token)
        assert caught.value.message == EMPTY_ANSWER_MESSAGE
        assert received == []  # no note above an answer that never came

    @pytest.mark.asyncio
    async def test_empty_summary_raises_the_retry_message(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [_chunk("Some text.", 0)]
        _, stream = await _service(store, self._silent_llm()).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        with pytest.raises(LLMProviderError) as caught:
            _ = [token async for token in stream]
        assert caught.value.message == EMPTY_ANSWER_MESSAGE

    @pytest.mark.asyncio
    async def test_overview_allows_the_answer_token_limit(self):
        # Reasoning counts toward max_tokens; the summary's 700 is too tight.
        store = AsyncMock()
        store.list_document_chunks.return_value = [_chunk("Some text.", 0)]
        llm = _mock_llm()
        _, stream = await _service(store, llm).prepare_overview_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="Give a summary",
        )
        _ = [token async for token in stream]
        config = llm.stream.call_args.args[1]
        assert config.max_tokens == Settings().llm_max_tokens


class TestMapReduce:
    @pytest.mark.asyncio
    async def test_medium_doc_streams_single_pass_below_context_budget(self):
        # A doc larger than the per-segment map budget but within the
        # context budget must NOT map — it streams in one pass. (Before the
        # single-pass/map split this mapped on the smaller map budget,
        # making medium docs needlessly slow.)
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk("word " * 200, i) for i in range(12)
        ]
        llm = _mock_llm()
        outcome, stream = await _service(
            store, llm, context_budget=10_000, map_token_budget=500
        ).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        [t async for t in stream]

        assert outcome.map_calls == 0
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_timeout_passed_to_map_calls(self):
        # Map calls (now streamed) must carry the (longer) summary timeout,
        # not the chat one. The returned stream isn't consumed here, so every
        # recorded stream() call is a map call.
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk("word " * 400, i) for i in range(6)
        ]
        llm = _mock_llm()
        await _service(
            store,
            llm,
            context_budget=500,
            map_token_budget=500,
            timeout_seconds=123,
        ).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        assert llm.stream.call_count > 0
        assert all(c.kwargs["timeout"] == 123 for c in llm.stream.call_args_list)

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
        # Map calls + 1 final exec-summary pass, all streamed.
        assert llm.stream.call_count == outcome.map_calls + 1

    @pytest.mark.asyncio
    async def test_map_calls_stream_and_feed_bullets_to_final_pass(self):
        # Each map call streams its bullets in several tokens; the joined,
        # stripped text of every call must reach the final pass, and the
        # non-streaming generate() must not be used at all.
        store = AsyncMock()
        store.list_document_chunks.return_value = [
            _chunk("word " * 400, i) for i in range(6)
        ]
        llm = AsyncMock()
        calls = {"n": 0}

        async def _stream_impl(messages, config=None, *, timeout=None):
            calls["n"] += 1
            for tok in [" - fact ", f"{calls['n']}", " \n"]:
                yield tok

        llm.stream = MagicMock(side_effect=_stream_impl)
        outcome, stream = await _service(
            store, llm, context_budget=500, map_token_budget=500
        ).prepare_summary_stream(
            workspace_id=uuid.uuid4(),
            user=_make_user(),
            documents=[("doc-1", "report.pdf")],
            query="summarize",
        )
        [t async for t in stream]

        llm.generate.assert_not_called()
        final_user_message = llm.stream.call_args_list[-1].args[0][1].content
        expected = "\n\n".join(f"- fact {n}" for n in range(1, outcome.map_calls + 1))
        assert final_user_message.endswith(f"MATERIAL:\n{expected}")

    @pytest.mark.asyncio
    async def test_chunk_truncation_guard(self):
        store = AsyncMock()
        store.list_document_chunks.return_value = [_chunk("text", i) for i in range(10)]
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


class TestDefaultBudgets:
    # Ollama's default num_ctx. Prompt plus generated tokens must fit it, or
    # the call is silently truncated.
    OLLAMA_DEFAULT_CTX = 4096

    @staticmethod
    def _default(name: str) -> int:
        return Settings.model_fields[name].default

    def test_map_call_fits_default_context(self):
        used = (
            self._default("summary_map_token_budget")
            + count_tokens(_MAP_SYSTEM_PROMPT)
            + self._default("summary_section_max_tokens")
        )
        assert used <= self.OLLAMA_DEFAULT_CTX

    def test_final_call_fits_default_context(self):
        query_allowance = 100
        used = (
            self._default("summary_context_token_budget")
            + count_tokens(_FINAL_SYSTEM_PROMPT)
            + query_allowance
            + self._default("summary_final_max_tokens")
        )
        assert used <= self.OLLAMA_DEFAULT_CTX


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
