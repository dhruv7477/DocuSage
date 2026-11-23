"""
tests/unit/test_llm_require_text.py
────────────────────────────────────
A reasoning model (gpt-oss) can spend its whole token budget on reasoning
and return no text. `require_text` turns that into a clear retry message.
"""

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from src.core.exceptions import LLMProviderError
from src.providers.llm.base import EMPTY_ANSWER_MESSAGE, require_text

pytestmark = pytest.mark.unit


async def _tokens(*tokens: str) -> AsyncIterator[str]:
    for token in tokens:
        yield token


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [token async for token in stream]


class TestRequireText:
    @pytest.mark.asyncio
    async def test_passes_tokens_through(self):
        tokens = await _drain(
            require_text(_tokens("Hello", " world."), purpose="answer")
        )
        assert tokens == ["Hello", " world."]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tokens", [(), ("", "\n", "  ")])
    async def test_no_visible_text_raises_the_retry_message(self, tokens):
        with pytest.raises(LLMProviderError) as caught:
            await _drain(require_text(_tokens(*tokens), purpose="answer"))
        assert (
            caught.value.message
            == EMPTY_ANSWER_MESSAGE
            == ("That didn't work, please try again.")
        )

    @pytest.mark.asyncio
    async def test_an_empty_answer_is_logged(self):
        with patch("src.providers.llm.base.log") as log:
            with pytest.raises(LLMProviderError):
                await _drain(require_text(_tokens(), purpose="overview"))
        log.warning.assert_called_once_with("llm_empty_answer", purpose="overview")
