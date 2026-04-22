"""
src/services/rag_service.py
────────────────────────────
RAG (Retrieval-Augmented Generation) service.

This is the core intelligence of the system — it answers user questions
by retrieving relevant document chunks and passing them to the LLM.

Full pipeline per query:
    1. Embed the user query (same model used during ingestion)
    2. Search Qdrant with user_id + workspace_id isolation filter
    3. Build prompt from retrieved chunks + chat history
    4. Call LLM (blocking or streaming)
    5. Return answer with source citations

Design principles:
- user_id isolation is ENFORCED in the SearchFilter — callers cannot
  bypass it. The vector search will only return that user's documents.
- Streaming is first-class: stream() is the primary path for chat UI.
- generate() exists for internal use (query rewriting, eval pipeline).
- Source citations are extracted from payloads — no second DB lookup needed
  because the text and metadata are stored in Qdrant payload.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.configs.config import get_settings
from src.core.exceptions import LLMProviderError, VectorStoreError
from src.core.logger import get_logger
from src.models.user import User
from src.providers.embedding.base import BaseEmbeddingProvider
from src.providers.llm.base import BaseLLMProvider, LLMConfig
from src.retrieval.pipeline import RetrievalPipeline
from src.services.prompt_builder import PromptBuilder
from src.vector_store.base import BaseVectorStore, SearchFilter, SearchResult

log = get_logger(__name__)
settings = get_settings()


@dataclass
class CitedSource:
    """A document chunk that contributed to the answer."""
    document_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete non-streaming response from the RAG pipeline."""
    answer: str
    sources: list[CitedSource] = field(default_factory=list)
    context_chunks_used: int = 0
    query_tokens_estimated: int = 0


class RAGService:
    """
    Retrieval-Augmented Generation service.

    All dependencies are injected — fully testable with mocks.
    Use the factory function get_rag_service() for production instances.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        llm: BaseLLMProvider,
        prompt_builder: PromptBuilder | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._top_k = top_k or settings.rag_top_k
        self._score_threshold = score_threshold or settings.rag_score_threshold
        self._pipeline = retrieval_pipeline

    # ── Non-Streaming Generate ────────────────────────────────────

    async def generate(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None = None,
        document_ids: list[str] | None = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline — returns complete answer.
        Use for non-chat contexts (evaluation, rewriting, admin tools).
        """
        search_results = await self._retrieve(
            query, workspace_id, user, document_ids
        )
        if not search_results:
            raise VectorStoreError(
                "No relevant documents found for this query.",
                detail="Upload and index documents before querying, or try a different question.",
            )
        prompt = self._prompt_builder.build(
            query=query,
            search_results=search_results,
            history=history_messages,
        )

        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        try:
            llm_response = await self._llm.generate(
                prompt.messages, llm_config
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM generation failed.", detail=str(exc)
            ) from exc

        sources = _extract_sources(search_results)

        log.info(
            "rag_generate",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            chunks_used=prompt.context_chunks_used,
            sources=len(sources),
            tokens=llm_response.total_tokens,
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            context_chunks_used=prompt.context_chunks_used,
            query_tokens_estimated=prompt.estimated_tokens,
        )

    # ── Streaming Generate ────────────────────────────────────────

    async def stream(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None = None,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming RAG pipeline — yields text tokens as they arrive.

        Usage in FastAPI SSE endpoint:
            async for token in rag.stream(query, workspace_id, user):
                yield f"data: {token}\\n\\n"

        The sources are sent as a final SSE event after the stream ends.
        """
        search_results = await self._retrieve(
            query, workspace_id, user, document_ids
        )
        if not search_results:
            raise VectorStoreError(
                "No relevant documents found for this query.",
                detail="Upload and index documents before querying, or try a different question.",
            )
        prompt = self._prompt_builder.build(
            query=query,
            search_results=search_results,
            history=history_messages,
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
        )

        try:
            async for token in self._llm.stream(prompt.messages, llm_config):
                yield token
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM streaming failed.", detail=str(exc)
            ) from exc

    async def get_sources_for_query(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        document_ids: list[str] | None = None,
    ) -> list[CitedSource]:
        """
        Retrieve sources without generating an answer.
        Used to pre-fetch citations for the streaming response.
        """
        results = await self._retrieve(query, workspace_id, user, document_ids)
        return _extract_sources(results)

    # ── Private ───────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        document_ids: list[str] | None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks via the advanced pipeline (if configured)
        or fall back to simple vector search.
        user_id isolation is ALWAYS enforced — it cannot be skipped.
        """
        if self._pipeline is not None:
            try:
                pipeline_result = await self._pipeline.retrieve(
                    query=query,
                    workspace_id=workspace_id,
                    user=user,
                    document_ids=document_ids,
                )
                return pipeline_result.results
            except Exception as exc:
                log.warning("pipeline_failed_falling_back", error=str(exc))
                # Fall through to simple search

        # Simple vector search (fallback / default)
        query_vector = await self._embedder.embed_query(query)

        search_filter = SearchFilter(
            user_id=str(user.id),
            workspace_id=str(workspace_id),
            document_ids=document_ids or [],
        )

        try:
            results = await self._vector_store.search(
                workspace_id=str(workspace_id),
                query_vector=query_vector,
                search_filter=search_filter,
                top_k=self._top_k,
                score_threshold=self._score_threshold,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "Vector search failed.", detail=str(exc)
            ) from exc

        log.debug(
            "vector_search_done",
            query_preview=query[:50],
            results=len(results),
            user_id=str(user.id),
        )
        return results


def _extract_sources(results: list[SearchResult]) -> list[CitedSource]:
    """
    Convert search results to CitedSource objects for the API response.
    Deduplicates by document_id + page_number (keeps highest score).
    """
    seen: dict[str, CitedSource] = {}

    for result in results:
        p = result.payload
        key = f"{p.document_id}:{p.page_number}"
        if key not in seen or result.score > seen[key].relevance_score:
            seen[key] = CitedSource(
                document_id=p.document_id,
                source_filename=p.source,
                page_number=p.page_number,
                chunk_index=p.chunk_index,
                relevance_score=round(result.score, 4),
            )

    return sorted(seen.values(), key=lambda s: s.relevance_score, reverse=True)
