"""
src/workers/tasks.py
──────────────────────
Celery task definitions.

Tasks bridge the sync Celery world with async Python.
We use asyncio.run() to execute async pipeline code from
within a synchronous Celery task.

ingest_document_task:
    Triggered by POST /documents/upload
    Runs the full ingestion pipeline for one document.

delete_document_vectors_task:
    Triggered by DELETE /documents/{id}
    Removes all vectors for the deleted document from Qdrant.
"""

import asyncio

from src.workers.celery_app import celery_app
from src.core.logger import get_logger
from src.ingestion.pipeline import IngestionResult

log = get_logger(__name__)


@celery_app.task(
    name="src.workers.tasks.ingest_document_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def ingest_document_task(self, document_id: str) -> dict:
    """
    Full ingestion pipeline for one document.
    Retries up to 3 times with 60-second delays on transient failures.
    """
    log.info("task_ingest_started", document_id=document_id)
    try:
        result = asyncio.run(_run_ingestion(document_id))
        log.info(
            "task_ingest_completed",
            document_id=document_id,
            chunks=result.chunks_created,
            status=result.status,
        )
        return {
            "document_id": document_id,
            "chunks": result.chunks_created,
            "status": result.status.value,
        }
    except Exception as exc:
        log.error("task_ingest_error", document_id=document_id, error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    name="src.workers.tasks.delete_document_vectors_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def delete_document_vectors_task(
    self, workspace_id: str, document_id: str, user_id: str
) -> dict:
    """Remove all Qdrant vectors for a deleted document."""
    log.info("task_vector_delete_started", document_id=document_id)
    try:
        asyncio.run(_run_vector_delete(workspace_id, document_id, user_id))
        return {"document_id": document_id, "deleted": True}
    except Exception as exc:
        log.error("task_vector_delete_error", document_id=document_id, error=str(exc))
        raise self.retry(exc=exc)


# ── Async Implementations ─────────────────────────────────────────

async def _run_ingestion(document_id: str) -> IngestionResult:  # noqa: F821
    """
    Async implementation of the ingestion pipeline.
    Creates its own DB session — Celery workers are separate processes.
    """
    from uuid import UUID
    from sqlalchemy import select

    from src.db.session import AsyncSessionLocal
    from src.ingestion.pipeline import IngestionPipeline
    from src.models.document import Document
    from src.providers.embedding.factory import get_embedding_provider
    from src.services.storage_service import StorageService
    from src.vector_store.factory import get_vector_store
    from src.configs.config import get_settings

    settings = get_settings()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.id == UUID(document_id))
        )
        document = result.scalar_one_or_none()
        if not document:
            raise ValueError(f"Document {document_id} not found.")

        pipeline = IngestionPipeline(
            embedder=get_embedding_provider(),
            vector_store=get_vector_store(),
            storage=StorageService(),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        ingestion_result = await pipeline.run(document, db)
        await db.commit()
        return ingestion_result


async def _run_vector_delete(
    workspace_id: str, document_id: str, user_id: str
) -> None:
    from src.vector_store.factory import get_vector_store

    vs = get_vector_store()
    await vs.delete_document_vectors(workspace_id, document_id, user_id)