"""
src/configs/config.py
─────────────
Central configuration using Pydantic BaseSettings.
All values are loaded from environment variables / .env file.
This is the ONLY place configuration is read — no os.getenv() elsewhere.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import AnyHttpUrl, Field, PostgresDsn, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────────
    app_name: str = "DocuSage"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_v1_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:3000"]

    # ── Security ────────────────────────────────────────────────
    secret_key: str = Field(min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── Database ────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "docusage"
    postgres_user: str = "docusage_user"
    postgres_password: str

    @computed_field  # type: ignore[misc]
    @property
    def database_url(self) -> str:
        """Async DSN constructed from parts — prevents misconfiguration."""
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def database_url_sync(self) -> str:
        """Sync DSN for Alembic migrations."""
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Vector DB ────────────────────────────────────────────────
    vector_store: Literal["qdrant", "pinecone", "weaviate"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "privateai"

    # ── Storage ──────────────────────────────────────────────────
    storage_backend: Literal["minio", "s3"] = "minio"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "privateai-docs"
    minio_secure: bool = False
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "privateai-docs"

    # ── LLM ──────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "vllm", "openai", "anthropic"] = "ollama"
    llm_model: str = "llama3.3:70b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 120
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "token-vllm"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # ── Embeddings ───────────────────────────────────────────────
    embedding_provider: Literal["bge", "ollama", "openai", "cohere"] = "bge"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 1024
    ollama_embed_model: str = "nomic-embed-text"
    openai_embed_model: str = "text-embedding-3-large"

    # ── RAG ──────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_score_threshold: float = 0.5
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Phase 4: Advanced Retrieval ──────────────────────────────
    # Query rewriting strategy: none | hyd | multi
    rewriter_strategy: str = "hyd"
    # Reranker backend: none | cross_encoder | llm
    reranker: str = "cross_encoder"
    # Cross-encoder model (sentence-transformers)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Candidates to fetch before reranking (should be > rag_top_k)
    retrieval_candidate_k: int = 20
    # Enable BM25 hybrid search alongside vector search
    enable_bm25: bool = True
    # Similarity threshold for near-duplicate deduplication
    dedup_threshold: float = 0.85

    # ── OAuth ────────────────────────────────────────────────────
    google_client_id: str = ""

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_per_minute: int = 60
    rate_limit_chat_per_minute: int = 20

    # ── Observability ────────────────────────────────────────────
    sentry_dsn: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Email ────────────────────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    emails_from: str = "noreply@privateai.com"

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce stricter rules in production to catch misconfigurations early."""
        if self.app_env == "production":
            if self.secret_key == "your-super-secret-key-change-in-production-min-32-chars":
                raise ValueError("SECRET_KEY must be changed in production!")
            if self.app_debug:
                raise ValueError("APP_DEBUG must be False in production!")
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Use this everywhere: `from src.configs.config import get_settings; settings = get_settings()`
    The lru_cache ensures .env is read exactly once at startup.
    """
    return Settings()