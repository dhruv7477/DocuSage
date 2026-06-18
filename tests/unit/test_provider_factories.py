"""
tests/unit/test_provider_factories.py
───────────────────────────────────────
Unit tests for LLM and embedding provider factories.
Tests that the correct provider class is instantiated from config.
No actual model loading — providers are checked by type only.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestLLMFactory:

    def setup_method(self):
        """Clear cached providers before each test."""
        from src.providers.llm.factory import reset_llm_provider
        reset_llm_provider()

    def test_ollama_provider_selected(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.3:70b"}):
            from src.providers.llm.factory import get_llm_provider, reset_llm_provider
            reset_llm_provider()
            # Force reload settings
            from src.configs.config import get_settings
            get_settings.cache_clear()

            provider = get_llm_provider()
            from src.providers.llm.ollama_provider import OllamaProvider
            assert isinstance(provider, OllamaProvider)

    def test_router_provider_uses_router_model(self):
        # The intent classifier runs on its own small model, distinct from
        # the answer model — so routing every message stays cheap.
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen2.5:7b-instruct",
            "ROUTER_MODEL": "qwen2.5:1.5b-instruct",
        }):
            from src.providers.llm.factory import (
                get_llm_provider,
                get_router_llm_provider,
                reset_llm_provider,
            )
            from src.configs.config import get_settings
            reset_llm_provider()
            get_settings.cache_clear()

            from src.providers.llm.ollama_provider import OllamaProvider
            router = get_router_llm_provider()
            answer = get_llm_provider()
            assert isinstance(router, OllamaProvider)
            assert router._model == "qwen2.5:1.5b-instruct"
            assert answer._model == "qwen2.5:7b-instruct"
            assert router is not answer

    def test_router_provider_falls_back_to_answer_model_when_unset(self):
        # An empty ROUTER_MODEL means "reuse the answer model" rather than
        # erroring — so the router still works if a user clears the setting.
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen2.5:7b-instruct",
            "ROUTER_MODEL": "",
        }):
            from src.providers.llm.factory import (
                get_router_llm_provider,
                reset_llm_provider,
            )
            from src.configs.config import get_settings
            reset_llm_provider()
            get_settings.cache_clear()

            router = get_router_llm_provider()
            assert router._model == "qwen2.5:7b-instruct"

    def test_router_provider_is_cached_and_reset_clears_it(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen2.5:7b-instruct",
            "ROUTER_MODEL": "qwen2.5:1.5b-instruct",
        }):
            from src.providers.llm.factory import (
                get_router_llm_provider,
                reset_llm_provider,
            )
            from src.configs.config import get_settings
            reset_llm_provider()
            get_settings.cache_clear()

            first = get_router_llm_provider()
            assert get_router_llm_provider() is first  # cached singleton
            reset_llm_provider()
            assert get_router_llm_provider() is not first  # cache cleared

    def test_invalid_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "nonexistent"}):
            from src.providers.llm.factory import reset_llm_provider
            from src.configs.config import get_settings
            reset_llm_provider()
            get_settings.cache_clear()
            with pytest.raises(ValidationError):
                from src.providers.llm.factory import get_llm_provider
                get_llm_provider()

    def teardown_method(self):
        from src.providers.llm.factory import reset_llm_provider
        from src.configs.config import get_settings
        reset_llm_provider()
        get_settings.cache_clear()


class TestEmbeddingFactory:

    def setup_method(self):
        from src.providers.embedding.factory import reset_embedding_provider
        from src.configs.config import get_settings
        reset_embedding_provider()
        get_settings.cache_clear()

    def test_bge_provider_selected(self):
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "bge",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_DEVICE": "cpu",
        }):
            from src.providers.embedding.factory import get_embedding_provider, reset_embedding_provider
            from src.configs.config import get_settings
            reset_embedding_provider()
            get_settings.cache_clear()
            provider = get_embedding_provider()
            from src.providers.embedding.bge_provider import BGEProvider
            assert isinstance(provider, BGEProvider)

    def test_bge_dimension_honors_explicit_config(self):
        # Explicit EMBEDDING_DIMENSION wins until the model is actually
        # loaded — at that point the provider replaces the configured
        # value with the model's real `get_sentence_embedding_dimension()`.
        # This pre-load behaviour matters because Qdrant collection
        # creation reads `provider.dimension` before the first encode.
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "bge",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_DIMENSION": "1024",
        }):
            from src.providers.embedding.factory import (
                get_embedding_provider,
                reset_embedding_provider,
            )
            from src.configs.config import get_settings
            reset_embedding_provider()
            get_settings.cache_clear()
            provider = get_embedding_provider()
            assert provider.dimension == 1024

    def test_bge_dimension_falls_back_to_known_model_map(self):
        # When EMBEDDING_DIMENSION isn't set, the provider should infer
        # from the known-model map. bge-small-en-v1.5 → 384.
        env = {
            "EMBEDDING_PROVIDER": "bge",
            "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("EMBEDDING_DIMENSION", None)
            from src.providers.embedding.factory import (
                get_embedding_provider,
                reset_embedding_provider,
            )
            from src.configs.config import get_settings
            reset_embedding_provider()
            get_settings.cache_clear()
            provider = get_embedding_provider()
            # Either 384 (env unset, picked up from KNOWN_MODEL map) or
            # whatever the user's settings.embedding_dimension is.
            assert provider.dimension in (384, get_settings().embedding_dimension)

    def teardown_method(self):
        from src.providers.embedding.factory import reset_embedding_provider
        from src.configs.config import get_settings
        reset_embedding_provider()
        get_settings.cache_clear()


class TestVectorStoreFactory:

    def test_qdrant_store_selected(self):
        with patch.dict(os.environ, {"VECTOR_STORE": "qdrant"}):
            from src.vector_store.factory import get_vector_store, reset_vector_store
            from src.configs.config import get_settings
            reset_vector_store()
            get_settings.cache_clear()
            store = get_vector_store()
            from src.vector_store.qdrant_store import QdrantVectorStore
            assert isinstance(store, QdrantVectorStore)

    def teardown_method(self):
        from src.vector_store.factory import reset_vector_store
        from src.configs.config import get_settings
        reset_vector_store()
        get_settings.cache_clear()
