"""
src/ingestion/pipeline.py
──────────────────────────
Document ingestion pipeline orchestrator.

Pipeline stages, mapped to `DocumentStatus` so the UI shows the user
exactly where a document is:

    PARSING   → Storage download + parser run
    CHUNKING  → Splitting parsed pages into RAG chunks
    EMBEDDING → Computing embeddings (usually the slowest stage)
    INDEXING  → Upserting vectors into Qdrant
    COMPLETED → Ready to query
    FAILED    → Terminal error; see `error_message`

Idempotency
-----------
Re-running ingestion for an existing document deletes its prior vectors
from Qdrant before upserting new ones. This keeps re-uploads (same
filename) clean even after retries or model swaps.

Error handling
--------------
Two error classes flow through this pipeline:

* Transient (`StorageConnectionError`, `ConnectionError`, `TimeoutError`,
  Ollama / Qdrant network blips) — re-raised so Celery's autoretry can
  fire. The document stays in the most recent in-flight status; on
  retry exhaustion the worker is responsible for the FAILED transition.

* Terminal (parse error, empty document, dimension mismatch) — caught
  here, the document is marked FAILED with a useful `error_message`,
  and the function returns an `IngestionResult` rather than re-raising.
  Retries on these errors don't help.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import (
    EmbeddingError,
    LLMProviderError,
    StorageError,
    VectorStoreError,
)
from src.core.logger import get_logger
from src.ingestion.chunker import SemanticChunker
from src.ingestion.parsers.registry import ParserRegistry
from src.models.document import Document, DocumentStatus
from src.providers.embedding.base import BaseEmbeddingProvider
from src.services.storage_service import StorageConnectionError, StorageService
from src.vector_store.base import BaseVectorStore, VectorPayload

log = get_logger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run (terminal outcome)."""

    document_id: str
    chunks_created: int
    status: DocumentStatus
    error: str | None = None


# Error types that signal a transient infrastructure problem. Re-raising
# these from `run()` lets Celery's autoretry policy take over instead of
# burning a permanent FAILED state on what is usually a network blip.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    StorageConnectionError,
    ConnectionError,
    TimeoutError,
)


class IngestionPipeline:
    """
    Orchestrates the full document ingestion pipeline.

    Injecting dependencies makes this trivially testable:
        pipeline = IngestionPipeline(mock_embedder, mock_store, mock_storage)
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        storage: StorageService,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._storage = storage
        # Map the legacy character-based settings onto the new
        # token-based semantic chunker. ~4 chars / token for English.
        approx_tokens = max(200, chunk_size // 4)
        approx_overlap = max(40, chunk_overlap // 4)
        if approx_overlap >= approx_tokens:
            approx_overlap = approx_tokens // 4
        self._chunker = SemanticChunker(
            chunk_tokens=approx_tokens,
            overlap_tokens=approx_overlap,
        )

    async def run(
        self,
        document: Document,
        db: AsyncSession,
    ) -> IngestionResult:
        """
        Execute the full ingestion pipeline for one document.

        On success: returns a COMPLETED IngestionResult.
        On terminal failure: marks document FAILED, returns FAILED result.
        On transient failure: re-raises so Celery can retry.
        """
        doc_id = str(document.id)
        log.info("ingestion_started", document_id=doc_id, filename=document.filename)

        try:
            # ── Stage 1: PARSING ──────────────────────────────────
            await self._set_progress(db, document, DocumentStatus.PARSING, 5)

            raw_bytes = await self._storage.download(document.storage_path)
            log.debug("file_downloaded", document_id=doc_id, bytes=len(raw_bytes))

            parser = ParserRegistry.get(document.filename)
            parsed_doc = await parser.parse(raw_bytes, document.filename)

            if parsed_doc.is_empty:
                # Terminal — no amount of retrying will produce text from
                # an empty file or a fully scanned PDF without OCR.
                raise _TerminalIngestionError(
                    f"Document '{document.filename}' produced no extractable text. "
                    "It may be a scanned image PDF (OCR is not yet enabled)."
                )

            # ── Stage 2: CHUNKING ─────────────────────────────────
            await self._set_progress(db, document, DocumentStatus.CHUNKING, 25)

            chunks = self._chunker.chunk_document(parsed_doc)
            if not chunks:
                raise _TerminalIngestionError("Chunking produced zero chunks.")

            log.debug("document_chunked", document_id=doc_id, chunks=len(chunks))

            # ── Stage 3: EMBEDDING ────────────────────────────────
            await self._set_progress(db, document, DocumentStatus.EMBEDDING, 45)

            # Use embedding_text() — prepends "heading > path \n\n" so
            # the embedder gets the chunk's structural context. This is
            # the single biggest retrieval-quality win in Phase 2.
            texts = [c.embedding_text() for c in chunks]
            vectors = await self._embedder.embed_documents(texts)

            # ── Stage 4: INDEXING ─────────────────────────────────
            await self._set_progress(db, document, DocumentStatus.INDEXING, 80)

            ws_id = str(document.workspace_id)
            if not await self._vector_store.collection_exists(ws_id):
                await self._vector_store.create_collection(
                    ws_id, dimension=self._embedder.dimension
                )
            else:
                # Idempotent re-index: drop any prior vectors for this
                # document_id so re-uploads / retries don't accumulate
                # stale chunks alongside fresh ones.
                try:
                    await self._vector_store.delete_document_vectors(
                        ws_id, doc_id, str(document.user_id)
                    )
                except Exception as exc:
                    # Non-fatal — worst case we end up with duplicates,
                    # not data loss. Log and continue.
                    log.warning(
                        "prior_vector_cleanup_failed",
                        document_id=doc_id,
                        error=str(exc),
                    )

            # Assign UUIDs first so we can stitch prev/next pointers
            # across the new vectors before they hit Qdrant.
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            payloads = [
                VectorPayload(
                    user_id=str(document.user_id),
                    workspace_id=ws_id,
                    document_id=doc_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    source=document.original_filename,
                    page_number=chunk.page_number,
                    chunk_total=len(chunks),
                    heading_path=list(chunk.heading_path),
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=chunk.token_count,
                    prev_chunk_id=chunk_ids[i - 1] if i > 0 else None,
                    next_chunk_id=chunk_ids[i + 1] if i < len(chunks) - 1 else None,
                )
                for i, chunk in enumerate(chunks)
            ]

            await self._vector_store.upsert_vectors(
                workspace_id=ws_id,
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=payloads,
            )

            # ── Stage 5: COMPLETED ────────────────────────────────
            document.status = DocumentStatus.COMPLETED
            document.chunk_count = len(chunks)
            document.error_message = None
            document.progress = 100
            await db.flush()

            log.info(
                "ingestion_completed",
                document_id=doc_id,
                chunks=len(chunks),
                filename=document.filename,
            )
            return IngestionResult(
                document_id=doc_id,
                chunks_created=len(chunks),
                status=DocumentStatus.COMPLETED,
            )

        except _TRANSIENT_ERRORS as exc:
            # Don't mark FAILED — Celery will retry. Just log and bubble.
            log.warning(
                "ingestion_transient_error",
                document_id=doc_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        except _TerminalIngestionError as exc:
            return await self._mark_failed(db, document, str(exc))
        except (
            VectorStoreError,
            EmbeddingError,
            LLMProviderError,
            StorageError,
        ) as exc:
            # Domain exceptions from providers — treat as terminal so the
            # user sees a useful message. Connection-class subtypes were
            # already caught above by _TRANSIENT_ERRORS.
            return await self._mark_failed(db, document, exc.message)
        except Exception as exc:
            # Anything else — treat as terminal but keep the type in logs.
            log.exception("ingestion_unexpected_error", document_id=doc_id)
            return await self._mark_failed(db, document, f"{type(exc).__name__}: {exc}")

    # ── Private ───────────────────────────────────────────────────

    async def _set_progress(
        self,
        db: AsyncSession,
        document: Document,
        status: DocumentStatus,
        progress: int,
    ) -> None:
        document.status = status
        document.progress = progress
        # Commit, not flush: the API reads status in its own transaction,
        # so an uncommitted stage is invisible until the whole run ends.
        await db.commit()

    async def _mark_failed(
        self,
        db: AsyncSession,
        document: Document,
        message: str,
    ) -> IngestionResult:
        log.error(
            "ingestion_failed",
            document_id=str(document.id),
            error=message,
        )
        document.status = DocumentStatus.FAILED
        document.error_message = message[:1000]
        document.progress = 0
        await db.flush()
        return IngestionResult(
            document_id=str(document.id),
            chunks_created=0,
            status=DocumentStatus.FAILED,
            error=message,
        )


class _TerminalIngestionError(Exception):
    """
    Internal marker for errors that shouldn't trigger a Celery retry.
    Examples: empty PDF, unparseable encoding, dimension mismatch.
    """
