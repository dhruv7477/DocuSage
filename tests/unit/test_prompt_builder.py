"""
tests/unit/test_prompt_builder.py
───────────────────────────────────
Unit tests for PromptBuilder — pure logic, no external deps.
"""

import pytest

from src.providers.llm.base import LLMMessage
from src.services.prompt_builder import (
    NO_ANSWER_PHRASE,
    PromptBuilder,
    _count_tokens,
)
from src.vector_store.base import SearchResult, VectorPayload


def _make_result(
    text: str,
    source: str = "doc.pdf",
    page: int = 1,
    score: float = 0.9,
    heading_path: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id="chunk-001",
        score=score,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id="doc-1",
            chunk_index=0,
            text=text,
            source=source,
            page_number=page,
            heading_path=heading_path or [],
        ),
    )


class TestPromptBuilder:

    def test_build_returns_system_user_messages(self):
        builder = PromptBuilder()
        result = builder.build(
            query="What is the policy?",
            search_results=[_make_result("The policy is X.")],
        )
        roles = [m.role for m in result.messages]
        assert "system" in roles
        assert "user" in roles

    def test_user_query_is_last_message(self):
        builder = PromptBuilder()
        result = builder.build(
            query="What is the policy?",
            search_results=[_make_result("Policy text.")],
        )
        assert result.messages[-1].role == "user"
        assert result.messages[-1].content == "What is the policy?"

    def test_context_injected_into_system_message(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Any question",
            search_results=[_make_result("Unique policy text 12345.")],
        )
        system_content = result.messages[0].content
        assert "Unique policy text 12345" in system_content

    def test_source_filename_in_context(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Question",
            search_results=[_make_result("Some text.", source="annual_report.pdf")],
        )
        assert "annual_report.pdf" in result.messages[0].content

    def test_heading_path_rendered_in_context(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Question",
            search_results=[_make_result(
                "Body text.",
                heading_path=["Chapter 3", "3.2 Refunds"],
            )],
        )
        # The header line should include the joined heading path.
        assert "Chapter 3 > 3.2 Refunds" in result.messages[0].content

    def test_empty_results_produces_no_context_message(self):
        builder = PromptBuilder()
        result = builder.build(query="Question", search_results=[])
        assert result.context_chunks_used == 0
        assert "no documents matched" in result.messages[0].content

    def test_grounded_prompt_enforces_citations(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Q", search_results=[_make_result("X.")],
        )
        sys_text = result.messages[0].content
        assert "[1]" in sys_text  # example citation form is shown
        assert NO_ANSWER_PHRASE in sys_text

    def test_abstention_swaps_to_no_answer_branch(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Q",
            search_results=[],
            abstain=True,
        )
        assert result.abstained is True
        assert NO_ANSWER_PHRASE in result.messages[0].content

    def test_history_injected_between_system_and_user(self):
        builder = PromptBuilder()
        history = [
            LLMMessage(role="user", content="Previous question"),
            LLMMessage(role="assistant", content="Previous answer"),
        ]
        result = builder.build(
            query="New question",
            search_results=[_make_result("Context.")],
            history=history,
        )
        # system → prev_user → prev_assistant → current_user
        assert len(result.messages) == 4
        assert result.messages[1].content == "Previous question"
        assert result.messages[2].content == "Previous answer"
        assert result.messages[3].content == "New question"

    def test_history_trimmed_by_token_budget(self):
        builder = PromptBuilder(max_history_tokens=10)  # Very small budget
        big_history = [
            LLMMessage(role="user", content="A " * 300),
            LLMMessage(role="assistant", content="B " * 300),
        ]
        result = builder.build(
            query="Q",
            search_results=[_make_result("C.")],
            history=big_history,
        )
        # Should have trimmed history — only system + current user
        assert result.history_messages_used == 0

    def test_duplicate_chunks_deduplicated(self):
        same_text = "Identical chunk text."
        results = [_make_result(same_text), _make_result(same_text)]
        builder = PromptBuilder()
        result = builder.build(query="Q", search_results=results)
        assert result.context_chunks_used == 1

    def test_chunks_counted_correctly(self):
        results = [
            _make_result("Chunk one."),
            _make_result("Chunk two."),
            _make_result("Chunk three."),
        ]
        builder = PromptBuilder()
        result = builder.build(query="Q", search_results=results)
        assert result.context_chunks_used == 3

    def test_estimated_tokens_positive(self):
        builder = PromptBuilder()
        result = builder.build(
            query="Test query",
            search_results=[_make_result("Some content.")],
        )
        assert result.estimated_tokens > 0


class TestCountTokens:
    def test_empty_string(self):
        assert _count_tokens("") >= 1

    def test_token_count_grows_with_text(self):
        a = _count_tokens("Hello world.")
        b = _count_tokens("Hello world. " * 50)
        assert b > a
