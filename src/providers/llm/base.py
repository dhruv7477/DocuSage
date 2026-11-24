"""
src/providers/llm/base.py
─────────────────────────
Abstract base class for all LLM providers.

Design:
- Defines the ONLY interface the rest of the application ever calls.
- Concrete providers (Ollama, vLLM, OpenAI) implement this contract.
- Swapping providers = changing one env var. Zero code changes elsewhere.

Methods:
    generate()  → full response as a string (for non-streaming use cases)
    stream()    → async generator of string chunks (for SSE chat responses)
    health()    → lightweight connectivity check (used at startup)

`require_text()` wraps a stream that must produce an answer for the user.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger

log = get_logger(__name__)

# Shown when a model returns no text at all. A reasoning model (gpt-oss) can
# spend its whole token budget on reasoning and stream nothing.
EMPTY_ANSWER_MESSAGE = "That didn't work, please try again."


async def require_text(
    tokens: AsyncIterator[str], *, purpose: str
) -> AsyncIterator[str]:
    """
    Pass a model's streamed tokens through; if none had visible text, log it
    and raise `LLMProviderError` with a retry message, so the user sees that
    instead of an empty answer. `purpose` names the stream in the log.
    """
    produced_text = False
    async for token in tokens:
        produced_text = produced_text or bool(token.strip())
        yield token
    if not produced_text:
        raise empty_answer(purpose)


def empty_answer(purpose: str) -> LLMProviderError:
    """Log an empty model answer and return the error that tells the user to retry."""
    log.warning("llm_empty_answer", purpose=purpose)
    return LLMProviderError(EMPTY_ANSWER_MESSAGE)


@dataclass
class LLMMessage:
    """
    A single message in a conversation.
    Mirrors the OpenAI chat format — all providers normalise to this.
    """

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    """
    Runtime generation parameters passed per-request.
    Defaults align with RAG use-cases (low temperature, bounded output).
    """

    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95
    stop_sequences: list[str] = field(default_factory=list)
    # "json" → provider-native structured output mode (Ollama `format: json`).
    # Providers that don't support it simply ignore the field.
    response_format: str | None = None


@dataclass
class LLMResponse:
    """Structured response from a generate() call."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLMProvider(ABC):
    """
    Abstract contract every LLM provider must satisfy.

    The application ONLY imports this class — never a concrete provider.
    Use get_llm_provider() from factory.py to get the active implementation.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> LLMResponse:
        """
        Send messages and return the complete response.
        Use for non-streaming calls (e.g. query rewriting, classification).

        `timeout` overrides the provider's default request timeout for this
        call only (e.g. summaries need longer than a chat turn). None =
        use the provider's configured default.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Send messages and yield response chunks as they arrive.
        Use for chat SSE endpoints where latency perception matters.

        `timeout` overrides the provider's default request timeout for this
        call only. None = use the provider's configured default.
        """
        ...

    @abstractmethod
    async def health(self) -> bool:
        """
        Returns True if the provider is reachable and responsive.
        Called during startup health checks.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier for logs."""
        ...
