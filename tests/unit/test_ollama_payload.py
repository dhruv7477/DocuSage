"""
tests/unit/test_ollama_payload.py
──────────────────────────────────
Pure payload-shape tests for OllamaProvider._build_payload — no HTTP.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.exceptions import LLMProviderError
from src.providers.llm.base import LLMConfig, LLMMessage
from src.providers.llm.ollama_provider import OllamaProvider

pytestmark = pytest.mark.unit


def _provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://localhost:11434", model="test-model")


MESSAGES = [LLMMessage(role="user", content="hi")]


class TestBuildPayload:
    def test_json_format_added_when_requested(self):
        payload = _provider()._build_payload(
            MESSAGES, LLMConfig(response_format="json"), stream=False
        )
        assert payload["format"] == "json"

    def test_no_format_key_by_default(self):
        payload = _provider()._build_payload(MESSAGES, LLMConfig(), stream=False)
        assert "format" not in payload

    def test_keep_alive_always_present(self):
        payload = _provider()._build_payload(MESSAGES, LLMConfig(), stream=True)
        assert payload["keep_alive"] == "30m"

    def test_options_carry_generation_params(self):
        config = LLMConfig(temperature=0.7, max_tokens=123, top_p=0.5)
        payload = _provider()._build_payload(MESSAGES, config, stream=False)
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 123
        assert payload["options"]["top_p"] == 0.5


class TestGenerateErrorMapping:
    """A timeout (model loading) must read differently from 'Ollama is down'."""

    async def test_timeout_maps_to_loading_message(self):
        provider = _provider()
        provider._client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("read timed out")
        )
        with pytest.raises(LLMProviderError) as exc:
            await provider.generate(MESSAGES)
        assert "timed out" in exc.value.message.lower()

    async def test_connect_error_maps_to_unreachable(self):
        provider = _provider()
        provider._client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(LLMProviderError) as exc:
            await provider.generate(MESSAGES)
        assert "reach ollama" in exc.value.message.lower()
