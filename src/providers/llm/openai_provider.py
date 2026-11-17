"""
src/providers/llm/openai_provider.py
──────────────────────────────────────
OpenAI LLM provider, and through `base_url` any OpenAI-compatible API
(Groq, Together, OpenRouter, a self-hosted vLLM server...).

Activation: LLM_PROVIDER=openai, OPENAI_API_KEY, OPENAI_MODEL, and
OPENAI_BASE_URL for a non-OpenAI endpoint, for example
https://api.groq.com/openai/v1.

Upstream failures become LLMProviderError with a message that is safe to
show to the user. A rate limit (HTTP 429), common on free tiers, becomes
LLMRateLimitedError after the SDK's own retries, which honour the
server's Retry-After header.
"""

from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from src.core.exceptions import LLMProviderError, LLMRateLimitedError
from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse

log = get_logger(__name__)

RATE_LIMITED_MESSAGE = (
    "The AI service is at its usage limit right now. Please try again in a minute."
)
TIMEOUT_MESSAGE = "The AI service took too long to answer. Please try again."


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = 60,
        base_url: str = "",
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        reasoning_effort: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")
        self._model = model
        # Sent only when set: reasoning models (gpt-oss on Groq, OpenAI o-series)
        # count their reasoning toward the token limits.
        self._reasoning_effort = reasoning_effort
        self._service = urlsplit(base_url).hostname if base_url else "api.openai.com"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def provider_name(self) -> str:
        return f"openai:{self._model}"

    def _provider_error(self, exc: Exception) -> LLMProviderError:
        """Map an OpenAI SDK error to a user-safe LLMProviderError."""
        log.warning(
            "llm_upstream_error",
            service=self._service,
            model=self._model,
            error_type=type(exc).__name__,
            status=getattr(exc, "status_code", None),
        )
        if isinstance(exc, RateLimitError):
            return LLMRateLimitedError(RATE_LIMITED_MESSAGE, detail=str(exc))
        if isinstance(exc, APITimeoutError):
            return LLMProviderError(TIMEOUT_MESSAGE, detail=str(exc))
        if isinstance(exc, APIConnectionError):
            return LLMProviderError(
                f"Cannot reach the AI service ({self._service}).", detail=str(exc)
            )
        if isinstance(exc, APIStatusError):
            return LLMProviderError(
                f"The AI service returned an error (HTTP {exc.status_code}).",
                detail=exc.message,
            )
        return LLMProviderError("The AI service failed.", detail=str(exc))

    def _request(self, cfg: LLMConfig) -> dict:
        extra: dict = {}
        if cfg.response_format == "json":
            extra["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            # extra_body: the pinned SDK predates the typed parameter.
            extra["extra_body"] = {"reasoning_effort": self._reasoning_effort}
        return extra

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
                **self._request(cfg),
            )
        except (APIConnectionError, APIStatusError) as exc:
            raise self._provider_error(exc) from exc

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
                stream=True,
                **self._request(cfg),
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except (APIConnectionError, APIStatusError) as exc:
            raise self._provider_error(exc) from exc

    async def health(self) -> bool:
        """True if the endpoint answers and lists the configured model.

        Lists the models rather than retrieving one: IDs such as
        "openai/gpt-oss-120b" contain a slash, which does not survive the
        /models/{id} path on every OpenAI-compatible server.
        """
        try:
            available = {m.id async for m in self._client.models.list()}
        except Exception as exc:
            log.warning(
                "llm_health_failed",
                service=self._service,
                model=self._model,
                error_type=type(exc).__name__,
            )
            return False
        if self._model not in available:
            log.warning(
                "llm_model_not_listed", service=self._service, model=self._model
            )
            return False
        return True
