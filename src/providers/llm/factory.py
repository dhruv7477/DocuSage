"""
src/providers/llm/factory.py
─────────────────────────────
LLM provider factory — the single point where config meets implementation.

Usage:
    from src.providers.llm.factory import get_llm_provider
    llm = get_llm_provider()         # returns BaseLLMProvider
    response = await llm.generate(messages)

The rest of the codebase NEVER imports a concrete provider class.
Switching providers is done exclusively via the LLM_PROVIDER env var.

Supported:
    ollama    → OllamaProvider   (default, local LLaMA via Ollama)
    vllm      → VLLMProvider     (production, GPU-accelerated)
    openai    → OpenAIProvider   (plug-in, GPT-4o etc.)
"""

from functools import lru_cache

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> BaseLLMProvider:
    """
    Returns the configured LLM provider as a cached singleton.

    lru_cache(maxsize=1) ensures the model/connection is initialised
    exactly once per process — critical for large models like LLaMA 70B.

    Raises:
        ValueError: if LLM_PROVIDER is set to an unsupported value.
    """
    settings = get_settings()
    provider = settings.llm_provider

    log.info("llm_provider_init", provider=provider, model=settings.llm_model)

    if provider == "ollama":
        from src.providers.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "vllm":
        from src.providers.llm.vllm_provider import VLLMProvider
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "openai":
        from src.providers.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Valid options: ollama | vllm | openai"
    )


def reset_llm_provider() -> None:
    """Clear the cached provider. Used in tests to swap providers."""
    get_llm_provider.cache_clear()