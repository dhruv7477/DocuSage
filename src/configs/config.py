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

from pydantic import Field, computed_field, model_validator
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
    # Documents in flight for longer than this are marked FAILED by the
    # stale-document sweep. Must exceed the longest legitimate wait
    # (queue time plus task_time_limit and retries).
    ingestion_stale_after_minutes: int = Field(default=60, ge=5)

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
    # Default is qwen2.5:7b-instruct (~4.7 GB Q4_K_M) — runs on CPU on a
    # laptop. The previous default (llama3.3:70b, ~40 GB) is unrealistic
    # for local dev and was the silent reason every chat call failed.
    llm_provider: Literal["ollama", "vllm", "openai", "anthropic"] = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 300
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "token-vllm"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # ── Embeddings ───────────────────────────────────────────────
    # Default is BGE-small-en-v1.5 (~133 MB, 384-dim) — fits comfortably
    # on CPU and gives strong English retrieval. BGE-M3 (1024-dim) is
    # available as an opt-in for multilingual / higher-quality use cases
    # but requires reindexing existing collections after the switch.
    embedding_provider: Literal["bge", "ollama", "openai", "cohere"] = "bge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384
    ollama_embed_model: str = "nomic-embed-text"
    openai_embed_model: str = "text-embedding-3-large"

    # ── RAG ──────────────────────────────────────────────────────
    # Recall-first defaults: pull a wide candidate pool with no score
    # threshold, then let the cross-encoder reranker decide. The
    # confidence-based abstention path (mean of top-3 reranker scores)
    # replaces the previous hard threshold filter.
    rag_top_k: int = 8
    rag_score_threshold: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Retrieval pipeline ───────────────────────────────────────
    # Query rewriting strategy in the legacy pipeline: none | hyd | multi.
    # Default flipped from "hyd" to "none": HyDE adds a 2–3s LLM round-trip
    # per query and on a local 7B model the recall lift rarely outweighs
    # the latency. The Phase 3 rewrite replaces this with a focused
    # standalone-question rewriter (only fires when chat history exists),
    # gated by `enable_hyde` for the opt-in HyDE path.
    rewriter_strategy: str = "none"
    enable_hyde: bool = False
    # Reranker backend: none | cross_encoder | llm
    reranker: str = "cross_encoder"
    # Default is ms-marco-MiniLM-L-6-v2 (~22 MB) — fast on CPU (~1-2s for 20
    # candidates). Upgrade to BAAI/bge-reranker-v2-m3 (~568 MB) for better
    # multilingual quality if you have a GPU; on CPU it scores 22 s/batch
    # and makes every query feel frozen.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Candidates fetched per retrieval branch before fusion+reranking.
    # Kept at 20 for CPU — the reranker is O(n) so fewer candidates = faster.
    retrieval_candidate_k: int = 20
    # Mean of top-3 reranker scores below this triggers an abstention
    # signal; the prompt builder swaps to a "graceful no-answer" branch.
    abstention_threshold: float = 0.3
    # Enable hybrid (dense + sparse) retrieval. The Phase 3 rewrite uses a
    # Qdrant full-text payload index; the legacy pipeline reuses this flag
    # for its (weaker) BM25-over-vector-results fallback.
    enable_hybrid_search: bool = True
    # Legacy alias kept until the Phase 3 retrieval rewrite removes the
    # old `RetrievalPipeline` that reads it.
    enable_bm25: bool = True
    # Near-duplicate Jaccard threshold used by the legacy dedup stage.
    dedup_threshold: float = 0.85

    # ── Agentic pipeline ─────────────────────────────────────────
    # Question Analyzer (intent router): off_topic | doc_summary |
    # rag_query. Kill switch for debugging — when false, every query
    # goes straight to the RAG path (pre-router behavior).
    enable_question_router: bool = True
    # Heuristics (regex greeting / summary detection) always run first and
    # are instant. The "is this about the documents?" decision is then made
    # by the RETRIEVAL SCORE, not an LLM: non-heuristic queries go to RAG,
    # and if retrieval finds nothing relevant (abstention) the orchestrator
    # declines. The reranker confidence measures doc-relevance directly and
    # is far more reliable than a small LLM classifier (which both answered
    # off-topic questions and wrongly refused real ones). Leave this False;
    # the LLM router (router_model) is an opt-in fallback only.
    router_use_llm: bool = False
    # Small, fast instruct model used ONLY for intent classification.
    # qwen2.5:1.5b-instruct is ~1 GB, same family as the answer model,
    # and reliable at JSON-mode classification. Override with ROUTER_MODEL.
    router_model: str = "qwen2.5:1.5b-instruct"
    # Fail open to rag_query if the router doesn't answer within this
    # budget. Generous so a cold router-model load doesn't fail open on
    # the first message; a warm 1.5B classify is only a few seconds.
    router_timeout_seconds: float = 30.0
    router_max_tokens: int = 150
    # Prompt token budgets. CPU prefill is the latency bottleneck on a
    # local 7B model — 2500 context tokens ≈ 10-25s faster than 4000.
    rag_context_tokens: int = 2500
    rag_history_tokens: int = 1000
    # DocSummaryService (map-reduce over a document's chunks).
    # Material up to this many tokens is summarized in ONE streamed pass;
    # larger docs are split into window-sized parts and map-reduced. Both
    # budgets are kept under the model context window (Ollama defaults to
    # num_ctx=4096) minus room for the prompt + generated tokens, so no
    # call silently truncates. Bigger docs split into more parts, but the
    # map phase runs those calls CONCURRENTLY (see summary_map_concurrency),
    # so wall-clock time stays bounded regardless of part count.
    summary_context_token_budget: int = 3200
    # Per-part input budget during the (parallel) map phase.
    summary_map_token_budget: int = 3200
    # Max in-flight map calls (asyncio.gather is bounded by this).
    # Default 1 = sequential, which is correct for CPU inference: parallel
    # LLM calls don't reduce wall-clock on a single CPU (the cores are
    # shared, so N concurrent calls each run ~Nx slower and tend to hit the
    # per-call timeout together). Raise this ONLY on a GPU / multi-slot
    # backend with spare compute, and set OLLAMA_NUM_PARALLEL >= this value.
    summary_map_concurrency: int = 1
    # Model used for summarization (bulk map-reduce over whole documents).
    # Empty → fall back to the lightweight router_model: summaries are
    # high-volume, low-precision work, and on a CPU/RAM-constrained box a
    # smaller model is far faster and uses much less memory than the answer
    # model. Set to llm_model (or any model) for higher-quality summaries
    # when you have the compute.
    summary_model: str = ""
    summary_section_max_tokens: int = 350
    summary_final_max_tokens: int = 700
    summary_max_total_chunks: int = 500
    # Summaries feed whole-document context to the model, so a single call
    # can run much longer than a chat turn — give it its own (larger)
    # timeout instead of the chat llm_timeout_seconds.
    summary_timeout_seconds: int = 600

    # ── OAuth ────────────────────────────────────────────────────
    google_client_id: str = ""

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_per_minute: int = 60
    rate_limit_chat_per_minute: int = 20

    # ── Observability ────────────────────────────────────────────
    sentry_dsn: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Startup readiness ────────────────────────────────────────
    # When true (default), the API refuses to start unless every
    # required dependency (Postgres, Qdrant, Redis, MinIO, Ollama with
    # the configured model pulled) is reachable. Set to false in tests
    # or when intentionally booting in a degraded mode.
    strict_startup: bool = True
    # Per-probe timeout. 15s is generous — picked because Windows
    # Docker Desktop's WSL2 network bridge can take 5–10s for the very
    # first connection to a containerized dep (cold DNS + TCP setup).
    # Once warm, real probes complete in <100ms; this only affects
    # cold-boot. The embedder probe gets its own larger budget (max(t, 30s))
    # because the first call may load a multi-hundred-MB model.
    startup_probe_timeout_seconds: float = 15.0

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
            if (
                self.secret_key
                == "your-super-secret-key-change-in-production-min-32-chars"
            ):
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
