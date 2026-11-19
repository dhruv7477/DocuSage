"""OpenAIProvider against a fake OpenAI-compatible server (httpx MockTransport)."""

import json

import httpx
import pytest

from src.core.exceptions import LLMProviderError, LLMRateLimitedError
from src.providers.llm.base import LLMConfig, LLMMessage
from src.providers.llm.openai_provider import (
    RATE_LIMITED_MESSAGE,
    TIMEOUT_MESSAGE,
    OpenAIProvider,
)

pytestmark = pytest.mark.unit

GROQ = "https://api.groq.com/openai/v1"
MESSAGES = [LLMMessage(role="user", content="hi")]


def _completion(text: str) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _provider(
    handler, max_retries: int = 2
) -> tuple[OpenAIProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    provider = OpenAIProvider(
        api_key="test-key",
        model="llama-test",
        base_url=GROQ,
        max_retries=max_retries,
        http_client=client,
    )
    return provider, seen


async def test_requests_go_to_the_configured_base_url() -> None:
    provider, seen = _provider(
        lambda req, n: httpx.Response(200, json=_completion("ok"))
    )

    resp = await provider.generate(MESSAGES)

    assert resp.content == "ok"
    assert str(seen[0].url) == f"{GROQ}/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"


async def test_json_mode_asks_for_a_json_object() -> None:
    provider, seen = _provider(
        lambda req, n: httpx.Response(200, json=_completion("{}"))
    )

    await provider.generate(MESSAGES, LLMConfig(response_format="json"))

    assert json.loads(seen[0].content)["response_format"] == {"type": "json_object"}


async def test_a_429_is_retried_and_then_succeeds() -> None:
    def handler(req: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(
                429, headers={"retry-after": "0"}, json={"error": {"message": "slow"}}
            )
        return httpx.Response(200, json=_completion("after retry"))

    provider, seen = _provider(handler)

    assert (await provider.generate(MESSAGES)).content == "after retry"
    assert len(seen) == 2


async def test_persistent_429_becomes_a_clear_rate_limit_error() -> None:
    provider, seen = _provider(
        lambda req, n: httpx.Response(
            429, headers={"retry-after": "0"}, json={"error": {"message": "slow"}}
        ),
        max_retries=1,
    )

    with pytest.raises(LLMRateLimitedError) as exc_info:
        await provider.generate(MESSAGES)

    assert exc_info.value.message == RATE_LIMITED_MESSAGE
    assert len(seen) == 2  # first try plus one retry


async def test_429_during_streaming_is_mapped_too() -> None:
    provider, _ = _provider(
        lambda req, n: httpx.Response(
            429, headers={"retry-after": "0"}, json={"error": {"message": "slow"}}
        ),
        max_retries=0,
    )

    with pytest.raises(LLMRateLimitedError):
        async for _ in provider.stream(MESSAGES):
            pass


async def test_timeouts_get_a_user_safe_message() -> None:
    def handler(req: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    provider, _ = _provider(handler, max_retries=0)

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(MESSAGES)

    assert exc_info.value.message == TIMEOUT_MESSAGE
