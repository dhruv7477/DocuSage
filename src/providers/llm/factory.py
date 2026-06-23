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
    # OpenAI uses its own model setting; ollama/vllm use llm_model.
    model = (
        settings.openai_model
        if settings.llm_provider == "openai"
        else settings.llm_model
    )
    return _build_llm_provider(model)


@lru_cache(maxsize=1)
def get_router_llm_provider() -> BaseLLMProvider:
    """
    Small, fast model dedicated to intent classification (the Question
    Analyzer routes every non-heuristic message through it).

    Kept separate from get_llm_provider() so classification runs on a
    ~1 GB model (router_model) instead of the 7B answer model — routing
    every message then costs a few seconds on CPU instead of ~90s.
    Falls back to the main answer model if router_model is unset.
    """
    settings = get_settings()
    return _build_llm_provider(settings.router_model or settings.llm_model)


@lru_cache(maxsize=1)
def get_summary_llm_provider() -> BaseLLMProvider:
    """
    Model for document summarization (bulk map-reduce over whole docs).

    Defaults to the lightweight router_model: summarization is high-volume
    work where a smaller, faster model is the right trade-off — especially
    on CPU/RAM-constrained hosts, where the full answer model's larger
    memory footprint and slower prefill cause timeouts. Override via
    SUMMARY_MODEL to use a bigger model when compute allows.
    """
    settings = get_settings()
    model = settings.summary_model or settings.router_model or settings.llm_model
    return _build_llm_provider(model)


def _build_llm_provider(model: str) -> BaseLLMProvider:
    """Construct a provider for the active backend pinned to `model`."""
    settings = get_settings()
    provider = settings.llm_provider

    log.info("llm_provider_init", provider=provider, model=model)

    if provider == "ollama":
        from src.providers.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=model,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "vllm":
        from src.providers.llm.vllm_provider import VLLMProvider
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=model,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "openai":
        from src.providers.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=model,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Valid options: ollama | vllm | openai"
    )


def reset_llm_provider() -> None:
    """Clear the cached providers. Used in tests to swap providers."""
    get_llm_provider.cache_clear()
    get_router_llm_provider.cache_clear()
    get_summary_llm_provider.cache_clear()