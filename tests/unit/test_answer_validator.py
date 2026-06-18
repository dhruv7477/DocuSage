"""
tests/unit/test_answer_validator.py
────────────────────────────────────
Pure-function tests for the programmatic answer validator.
"""

import pytest

from src.services.answer_validator import validate_answer
from src.services.prompt_builder import NO_ANSWER_PHRASE

pytestmark = pytest.mark.unit


class TestCitationStripping:

    def test_in_range_citations_kept(self):
        result = validate_answer(
            "The deadline is three days [1][2].",
            num_sources=3, abstained=False,
        )
        assert result.answer == "The deadline is three days [1][2]."
        assert not result.changed
        assert result.stripped_citations == []

    def test_out_of_range_citation_stripped(self):
        result = validate_answer(
            "The deadline is three days [1][7].",
            num_sources=3, abstained=False,
        )
        assert "[7]" not in result.answer
        assert "[1]" in result.answer
        assert result.changed
        assert result.stripped_citations == [7]

    def test_zero_citation_stripped(self):
        result = validate_answer("Fact [0].", num_sources=3, abstained=False)
        assert "[0]" not in result.answer
        assert result.stripped_citations == [0]

    def test_whitespace_cleanup_after_strip(self):
        result = validate_answer(
            "Deadline is three days [9] .",
            num_sources=2, abstained=False,
        )
        assert "  " not in result.answer
        assert " ." not in result.answer

    def test_citations_disallowed_strips_all(self):
        result = validate_answer(
            "Summary point [1] and detail [2].",
            num_sources=5, abstained=False, allow_citations=False,
        )
        assert "[1]" not in result.answer
        assert "[2]" not in result.answer
        assert result.stripped_citations == [1, 2]

    def test_idempotent_on_clean_answer(self):
        clean = "A perfectly normal answer with no citations."
        result = validate_answer(clean, num_sources=0, abstained=False)
        assert result.answer == clean
        assert not result.changed
        assert result.warning_code is None


class TestAbstentionContract:

    def test_abstained_answer_replaced_when_phrase_missing(self):
        result = validate_answer(
            "The capital of France is Paris.",
            num_sources=0, abstained=True,
        )
        assert result.answer == NO_ANSWER_PHRASE
        assert result.changed
        assert result.warning_code == "abstention_mismatch"

    def test_abstained_answer_kept_when_phrase_present(self):
        answer = f"{NO_ANSWER_PHRASE} Related topics in your documents include: visas."
        result = validate_answer(answer, num_sources=0, abstained=True)
        assert result.answer == answer
        assert result.warning_code is None

    def test_voluntary_abstention_flagged(self):
        result = validate_answer(
            NO_ANSWER_PHRASE,
            num_sources=4, abstained=False,
        )
        assert result.answer == NO_ANSWER_PHRASE
        assert result.warning_code == "model_abstained"
