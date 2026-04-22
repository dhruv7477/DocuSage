"""
src/providers/llm/openai_provider.py
──────────────────────────────────────
OpenAI LLM provider — plug-in for GPT-4o and other OpenAI models.

Activation: set LLM_PROVIDER=openai and OPENAI_API_KEY in .env.
Zero other code changes needed — same BaseLLMProvider interface.

Note: vLLMProvider is nearly identical because vLLM uses the OpenAI SDK.
This class exists as a dedicated OpenAI implementation that can handle
OpenAI-specific features (function calling, vision, etc.) in future phases.
"""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse

log = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):

    def __init__(self, api_key: str, model: str, timeout: int = 60) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=2)

    @property
    def provider_name(self) -> str:
        return f"openai:{self._model}"

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        cfg = config or LLMConfig()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stop=cfg.stop_sequences or None,
            )
        except APIConnectionError as exc:
            raise LLMProviderError("Cannot reach OpenAI API.", detail=str(exc)) from exc
        except APIStatusError as exc:
            raise LLMProviderError(
                f"OpenAI API error {exc.status_code}", detail=exc.message
            ) from exc

        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or LLMConfig()
        try:
            async with await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except APIConnectionError as exc:
            raise LLMProviderError("Cannot reach OpenAI during stream.", detail=str(exc)) from exc

    async def health(self) -> bool:
        try:
            await self._client.models.retrieve(self._model)
            return True
        except Exception:
            return False
