"""
tests/unit/test_query_understanding.py
──────────────────────────────────────
The clean-up QueryUnderstanding applies to the model's standalone rewrite:
wrapping quotes, list bullets and numbering come off; the question stays.
"""

from unittest.mock import AsyncMock

import pytest

from src.providers.llm.base import LLMResponse
from src.retrieval.query_understanding import QueryUnderstanding

pytestmark = pytest.mark.unit

HISTORY = [("user", "Tell me about annual leave."), ("assistant", "It is 20 days.")]
FOLLOW_UP = "Is it paid?"


async def _rewrite(model_reply: str) -> str:
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(content=model_reply, model="test")
    )
    return await QueryUnderstanding(llm)._rewrite_standalone(FOLLOW_UP, HISTORY)


@pytest.mark.parametrize(
    ("model_reply", "expected"),
    [
        ('"Is annual leave paid?"', "Is annual leave paid?"),
        ("`'Is annual leave paid?'`", "Is annual leave paid?"),
        ("- Is annual leave paid?", "Is annual leave paid?"),
        ("* Is annual leave paid?", "Is annual leave paid?"),
        ("1. Is annual leave paid?", "Is annual leave paid?"),
        ("2) Is annual leave paid?", "Is annual leave paid?"),
        ('  "Is annual leave paid?"  \n', "Is annual leave paid?"),
        ('What does "annual leave" cover?', 'What does "annual leave" cover?'),
    ],
)
async def test_strips_wrapping_from_the_rewrite(
    model_reply: str, expected: str
) -> None:
    assert await _rewrite(model_reply) == expected


async def test_keeps_a_trailing_dash_or_star() -> None:
    # Only quotes are stripped from the end: "-" and "*" are leading bullets.
    assert await _rewrite("Is annual leave paid -") == "Is annual leave paid -"


@pytest.mark.parametrize("model_reply", ['""', "``", "- 1.", "  "])
async def test_falls_back_to_the_question_when_nothing_is_left(
    model_reply: str,
) -> None:
    assert await _rewrite(model_reply) == FOLLOW_UP


async def test_leaves_a_long_run_of_inner_quotes_alone() -> None:
    reply = "Is " + '"' * 20_000 + " paid?"
    assert await _rewrite(reply) == reply


async def test_falls_back_to_the_question_when_the_model_fails() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("model down"))
    assert (
        await QueryUnderstanding(llm)._rewrite_standalone(FOLLOW_UP, HISTORY)
        == FOLLOW_UP
    )
