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

    def test_bge_dimension_is_correct(self):
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "bge",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
        }):
            from src.providers.embedding.factory import get_embedding_provider, reset_embedding_provider
            from src.configs.config import get_settings
            reset_embedding_provider()
            get_settings.cache_clear()
            provider = get_embedding_provider()
            assert provider.dimension == 1024

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
