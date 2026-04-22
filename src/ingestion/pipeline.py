"""
src/ingestion/pipeline.py
──────────────────────────
Document ingestion pipeline orchestrator.

Pipeline:
    1. Download raw bytes from storage (file already uploaded by API)
    2. Parse bytes → ParsedDocument (using ParserRegistry)
    3. Chunk ParsedDocument → list[TextChunk]
    4. Embed chunks in batches → list[list[float]]
    5. Build VectorPayload for each chunk
    6. Upsert vectors into Qdrant (workspace collection)
    7. Update document status in PostgreSQL

Design:
    This class coordinates all components but owns none of them.
    Each component (parser, chunker, embedder, vector store) is injected
    or retrieved from its factory — fully testable in isolation.

    Called by:
        - Celery worker task (async execution)
        - Direct await in tests

    On failure:
        Sets document.status = FAILED with error_message.
        Does NOT raise — the worker logs and moves on.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.parsers.registry import ParserRegistry
from src.models.document import Document, DocumentStatus
from src.providers.embedding.base import BaseEmbeddingProvider
from src.services.storage_service import StorageService
from src.vector_store.base import BaseVectorStore, SearchFilter, VectorPayload

log = get_logger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""
    document_id: str
    chunks_created: int
    status: DocumentStatus
    error: str | None = None


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
        self._chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def run(
        self,
        document: Document,
        db: AsyncSession,
    ) -> IngestionResult:
        """
        Execute the full ingestion pipeline for one document.
        Updates document.status throughout. Never raises — logs errors.
        """
        doc_id = str(document.id)
        log.info("ingestion_started", document_id=doc_id, filename=document.filename)

        try:
            # ── Step 1: Mark as processing ────────────────────────
            document.status = DocumentStatus.PROCESSING
            await db.flush()

            # ── Step 2: Download raw bytes from storage ───────────
            raw_bytes = await self._storage.download(document.storage_path)
            log.debug("file_downloaded", document_id=doc_id, bytes=len(raw_bytes))

            # ── Step 3: Parse bytes into structured pages ─────────
            parser = ParserRegistry.get(document.filename)
            parsed_doc = await parser.parse(raw_bytes, document.filename)

            if parsed_doc.is_empty:
                raise ValueError(
                    f"Document '{document.filename}' produced no extractable text. "
                    "It may be a scanned image PDF."
                )

            # ── Step 4: Chunk into RAG-ready pieces ───────────────
            chunks = self._chunker.chunk_document(parsed_doc)
            if not chunks:
                raise ValueError("Chunking produced zero chunks.")

            log.debug(
                "document_chunked", document_id=doc_id, chunks=len(chunks)
            )

            # ── Step 5: Embed all chunks ──────────────────────────
            texts = [c.text for c in chunks]
            vectors = await self._embedder.embed_documents(texts)

            # ── Step 6: Ensure Qdrant collection exists ───────────
            ws_id = str(document.workspace_id)
            if not await self._vector_store.collection_exists(ws_id):
                await self._vector_store.create_collection(
                    ws_id, dimension=self._embedder.dimension
                )

            # ── Step 7: Build payloads and upsert ────────────────
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
                )
                for chunk in chunks
            ]

            await self._vector_store.upsert_vectors(
                workspace_id=ws_id,
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=payloads,
            )

            # ── Step 8: Mark completed in PostgreSQL ─────────────
            document.status = DocumentStatus.COMPLETED
            document.chunk_count = len(chunks)
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

        except Exception as exc:
            log.error(
                "ingestion_failed",
                document_id=doc_id,
                error=str(exc),
            )
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)[:1000]
            await db.flush()
            return IngestionResult(
                document_id=doc_id,
                chunks_created=0,
                status=DocumentStatus.FAILED,
                error=str(exc),
            )