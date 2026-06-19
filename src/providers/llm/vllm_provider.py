"""
src/providers/llm/vllm_provider.py
────────────────────────────────────
vLLM provider — production-grade LLaMA serving.

vLLM exposes a fully OpenAI-compatible /v1/chat/completions endpoint,
so this uses the openai SDK pointed at your vLLM server.

Why vLLM for production:
- Continuous batching: serves multiple users simultaneously on one GPU
- PagedAttention: 2-5x higher throughput vs naive serving
- OpenAI-compatible: drop-in replacement, no SDK changes needed
- Supports LLaMA 3.3 70B, Mistral, Qwen, and most HuggingFace models

Setup:
    pip install vllm
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --port 8001
"""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse

log = get_logger(__name__)


class VLLMProvider(BaseLLMProvider):
    """
    LLM provider backed by a vLLM server (OpenAI-compatible).
    Uses the official openai SDK — same client code works for real OpenAI too.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,        # vLLM accepts any non-empty string
            timeout=timeout,
            max_retries=2,
        )

    @property
    def provider_name(self) -> str:
        return f"vllm:{self._model}"

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> LLMResponse:
        cfg = config or LLMConfig()
        client = self._client.with_options(timeout=timeout) if timeout else self._client
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stop=cfg.stop_sequences or None,
                stream=False,
            )
        except APIConnectionError as exc:
            raise LLMProviderError(
                "Cannot reach vLLM server.", detail=str(exc)
            ) from exc
        except APIStatusError as exc:
            raise LLMProviderError(
                f"vLLM returned error {exc.status_code}", detail=exc.message
            ) from exc

        choice = resp.choices[0]
        usage = resp.usage

        log.debug("vllm_generate", model=self._model, tokens=usage.total_tokens if usage else 0)
        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or LLMConfig()
        client = self._client.with_options(timeout=timeout) if timeout else self._client
        try:
            async with await client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stop=cfg.stop_sequences or None,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except APIConnectionError as exc:
            raise LLMProviderError("Cannot reach vLLM during stream.", detail=str(exc)) from exc
        except APIStatusError as exc:
            raise LLMProviderError(f"vLLM stream error {exc.status_code}") from exc

    async def health(self) -> bool:
        try:
            models = await self._client.models.list()
            return any(self._model in m.id for m in models.data)
        except Exception:
            return False
