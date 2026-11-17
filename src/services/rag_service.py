"""
src/services/rag_service.py
────────────────────────────
RAG (Retrieval-Augmented Generation) service.

Pipeline per query
------------------
  1. Query understanding (standalone-question rewrite if needed)
  2. Hybrid retrieval (dense + sparse, RRF-fused)
  3. Cross-encoder rerank → top_k chunks + confidence
  4. Prompt build (grounded-answer or abstention branch)
  5. Stream LLM tokens / persist sources

What changed vs. the legacy service
-----------------------------------
* No more hard `raise VectorStoreError("No relevant documents found")`.
  The Phase 4 prompt builder will use the abstention prompt when
  retrieval confidence is below threshold, so the LLM produces a
  graceful "I cannot find this in your documents." answer instead of
  the chat UI freezing on a generic error.
* No legacy "fall back to simple vector search" branch. Retrieval
  always goes through the new `HybridRetriever` + reranker so behavior
  is predictable and testable.
* `CitedSource` carries `heading_path`, `char_start`, `char_end`,
  `confidence` so the frontend can render rich citations and link to
  the exact passage in the original document.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from src.configs.config import get_settings
from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger
from src.models.user import User
from src.providers.embedding.base import BaseEmbeddingProvider
from src.providers.llm.base import BaseLLMProvider, LLMConfig
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_understanding import QueryEnrichment, QueryUnderstanding
from src.retrieval.reranker import BaseReranker, NoOpReranker, compute_confidence
from src.services.prompt_builder import BuiltPrompt, PromptBuilder
from src.vector_store.base import BaseVectorStore, SearchResult

log = get_logger(__name__)
settings = get_settings()


@dataclass
class CitedSource:
    """A document chunk that contributed to the answer."""

    document_id: str
    chunk_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    relevance_score: float
    heading_path: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    text: str = ""


@dataclass
class RAGResponse:
    """Complete non-streaming response from the RAG pipeline."""

    answer: str
    sources: list[CitedSource] = field(default_factory=list)
    context_chunks_used: int = 0
    confidence: float = 0.0
    abstained: bool = False
    prompt: BuiltPrompt | None = None


@dataclass
class _RetrievalOutcome:
    """Internal: results carried through to the prompt-building step."""

    results: list[SearchResult]
    confidence: float
    enrichment: QueryEnrichment
    # The prompt actually sent to the LLM. `prompt.included_results[n-1]`
    # is the chunk behind citation [n] — the orchestrator builds the
    # sources event from this, never from a re-sorted view.
    prompt: BuiltPrompt | None = None


class RAGService:
    """
    Retrieval-Augmented Generation service.

    All collaborators are injected so this is trivially testable with
    mocks (`tests/unit/test_rag_service.py`). For production wiring see
    `src.api.v1.chat._get_rag_service()`.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        llm: BaseLLMProvider,
        prompt_builder: PromptBuilder | None = None,
        retriever: HybridRetriever | None = None,
        reranker: BaseReranker | None = None,
        query_understanding: QueryUnderstanding | None = None,
        top_k: int | None = None,
        abstention_threshold: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._retriever = retriever or HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            candidate_k=settings.retrieval_candidate_k,
        )
        self._reranker = reranker or NoOpReranker()
        self._query_understanding = query_understanding or QueryUnderstanding(
            llm=llm,
            enable_hyde=settings.enable_hyde,
        )
        self._top_k = top_k or settings.rag_top_k
        self._abstention_threshold = (
            abstention_threshold
            if abstention_threshold is not None
            else settings.abstention_threshold
        )

    # ── Non-streaming ────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None = None,
        document_ids: list[str] | None = None,
    ) -> RAGResponse:
        outcome = await self._retrieve_and_rerank(
            query=query,
            workspace_id=workspace_id,
            user=user,
            history_messages=history_messages,
            document_ids=document_ids,
        )
        abstained = self._is_abstaining(outcome)

        prompt = self._prompt_builder.build(
            query=outcome.enrichment.standalone,
            search_results=outcome.results,
            history=history_messages,
            abstain=abstained,
        )

        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        try:
            llm_response = await self._llm.generate(prompt.messages, llm_config)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM generation failed.",
                detail=str(exc),
            ) from exc

        sources = _extract_sources(outcome.results)

        log.info(
            "rag_generate",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            chunks_used=prompt.context_chunks_used,
            sources=len(sources),
            confidence=outcome.confidence,
            abstained=abstained,
            tokens=llm_response.total_tokens,
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            context_chunks_used=prompt.context_chunks_used,
            confidence=outcome.confidence,
            abstained=abstained,
            prompt=prompt,
        )

    # ── Streaming ────────────────────────────────────────────────

    async def stream(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None = None,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens for the answer. Sources / abstention are not
        emitted from this iterator — the SSE endpoint reads them off
        `last_outcome` after the stream finishes.

        We deliberately stream tokens as bare strings (not SSE events)
        so this method stays testable without an HTTP layer; the
        endpoint wraps each token in the `{"type":"token",...}` envelope.
        """
        outcome = await self._retrieve_and_rerank(
            query=query,
            workspace_id=workspace_id,
            user=user,
            history_messages=history_messages,
            document_ids=document_ids,
        )
        self.last_outcome = outcome  # consumed by the SSE endpoint
        abstained = self._is_abstaining(outcome)

        prompt = self._prompt_builder.build(
            query=outcome.enrichment.standalone,
            search_results=outcome.results,
            history=history_messages,
            abstain=abstained,
        )

        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        log.info(
            "rag_stream_start",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            chunks_used=prompt.context_chunks_used,
            confidence=outcome.confidence,
            abstained=abstained,
        )

        try:
            async for token in self._llm.stream(prompt.messages, llm_config):
                yield token
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM streaming failed.",
                detail=str(exc),
            ) from exc

    async def prepare_stream(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None = None,
        document_ids: list[str] | None = None,
        *,
        on_stage: "Callable[[str], None] | None" = None,
    ) -> "tuple[_RetrievalOutcome, AsyncIterator[str]]":
        """
        Retrieve context and return (outcome, token_stream) together.

        Preferred over stream() for the SSE endpoint — the caller gets the
        retrieval outcome before any tokens are generated, eliminating the
        self.last_outcome instance-state side-channel.

        `on_stage` (optional) is called with "retrieving"/"reranking" as the
        pipeline advances so the SSE endpoint can surface live progress.
        """
        outcome = await self._retrieve_and_rerank(
            query=query,
            workspace_id=workspace_id,
            user=user,
            history_messages=history_messages,
            document_ids=document_ids,
            on_stage=on_stage,
        )
        abstained = self._is_abstaining(outcome)

        prompt = self._prompt_builder.build(
            query=outcome.enrichment.standalone,
            search_results=outcome.results,
            history=history_messages,
            abstain=abstained,
        )
        outcome.prompt = prompt

        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        log.info(
            "rag_stream_start",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            chunks_used=prompt.context_chunks_used,
            confidence=outcome.confidence,
            abstained=abstained,
        )

        async def _token_iter() -> AsyncIterator[str]:
            try:
                async for token in self._llm.stream(prompt.messages, llm_config):
                    yield token
            except LLMProviderError:
                raise
            except Exception as exc:
                raise LLMProviderError(
                    "LLM streaming failed.",
                    detail=str(exc),
                ) from exc

        return outcome, _token_iter()

    async def get_sources_for_query(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        document_ids: list[str] | None = None,
    ) -> list[CitedSource]:
        """Retrieve sources without invoking the LLM. Used for previews."""
        outcome = await self._retrieve_and_rerank(
            query=query,
            workspace_id=workspace_id,
            user=user,
            history_messages=None,
            document_ids=document_ids,
        )
        return _extract_sources(outcome.results)

    # ── Internals ────────────────────────────────────────────────

    async def _retrieve_and_rerank(
        self,
        *,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None,
        document_ids: list[str] | None,
        on_stage: "Callable[[str], None] | None" = None,
    ) -> _RetrievalOutcome:
        # Build the (role, content) tuples for query understanding.
        # Only the most recent ~4 turns matter; the prompt builder owns
        # full-history compression separately.
        history: list[tuple[str, str]] = []
        if history_messages:
            for msg in history_messages[-4:]:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)
                if role and content:
                    history.append((str(role), str(content)))

        log.info(
            "rag_stage_enrich",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            query_preview=query[:80],
        )
        enrichment = await self._query_understanding.enrich_query(
            query=query, history=history
        )

        log.info(
            "rag_stage_retrieve",
            workspace_id=str(workspace_id),
            rewritten=enrichment.rewritten,
        )
        if on_stage:
            on_stage("retrieving")
        retrieval = await self._retriever.retrieve(
            workspace_id=workspace_id,
            user_id=user.id,
            enrichment=enrichment,
            document_ids=document_ids,
        )

        log.info(
            "rag_stage_rerank",
            candidates=len(retrieval.results),
            top_k=self._top_k,
        )
        if on_stage:
            on_stage("reranking")
        # Rerank against the user's *original* wording so the cross-
        # encoder is judging answers to what the user actually asked.
        reranked = await self._reranker.rerank(
            query=enrichment.original,
            results=retrieval.results,
            top_k=self._top_k,
        )
        confidence = compute_confidence(reranked)

        log.info(
            "rag_stage_rerank_done",
            reranked=len(reranked),
            confidence=round(confidence, 3),
        )
        return _RetrievalOutcome(
            results=reranked,
            confidence=confidence,
            enrichment=enrichment,
        )

    def _is_abstaining(self, outcome: _RetrievalOutcome) -> bool:
        return not outcome.results or outcome.confidence < self._abstention_threshold


def _results_to_cited(results: list[SearchResult]) -> list[CitedSource]:
    """
    Convert search results to CitedSource objects preserving order —
    NO dedup, NO re-sort. Use with `BuiltPrompt.included_results` so
    that sources[n-1] is exactly the chunk behind citation [n].
    """
    return [
        CitedSource(
            document_id=r.payload.document_id,
            chunk_id=r.chunk_id,
            source_filename=r.payload.source,
            page_number=r.payload.page_number,
            chunk_index=r.payload.chunk_index,
            relevance_score=round(r.score, 4),
            heading_path=list(r.payload.heading_path),
            char_start=r.payload.char_start,
            char_end=r.payload.char_end,
            text=r.payload.text,
        )
        for r in results
    ]


def _extract_sources(results: list[SearchResult]) -> list[CitedSource]:
    """
    Convert search results to `CitedSource` objects.

    Deduplicates by `document_id + page_number` while keeping the
    highest score, then sorts by score descending. The chunk_id is
    preserved on the survivor so the UI can fetch the exact passage if
    we ever add a "view in original document" affordance.
    """
    seen: dict[str, CitedSource] = {}
    for result in results:
        p = result.payload
        key = f"{p.document_id}:{p.page_number}"
        new = CitedSource(
            document_id=p.document_id,
            chunk_id=result.chunk_id,
            source_filename=p.source,
            page_number=p.page_number,
            chunk_index=p.chunk_index,
            relevance_score=round(result.score, 4),
            heading_path=list(p.heading_path),
            char_start=p.char_start,
            char_end=p.char_end,
            text=p.text,
        )
        existing = seen.get(key)
        if existing is None or new.relevance_score > existing.relevance_score:
            seen[key] = new
    return sorted(seen.values(), key=lambda s: s.relevance_score, reverse=True)
