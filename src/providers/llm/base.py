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
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    """
    A single message in a conversation.
    Mirrors the OpenAI chat format — all providers normalise to this.
    """
    role: str       # "system" | "user" | "assistant"
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
    ) -> LLMResponse:
        """
        Send messages and return the complete response.
        Use for non-streaming calls (e.g. query rewriting, classification).
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Send messages and yield response chunks as they arrive.
        Use for chat SSE endpoints where latency perception matters.
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