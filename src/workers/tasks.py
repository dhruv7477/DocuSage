"""
src/workers/tasks.py
──────────────────────
Celery task definitions.

Tasks bridge the sync Celery world with async Python. We use
asyncio.run() to execute async pipeline code from a synchronous task.

Retry policy
------------
Only transient errors trigger retries — terminal errors (parse failure,
empty document, etc.) flow back as a FAILED `IngestionResult` without
an exception, which Celery treats as success and so won't retry. This
prevents the previous failure mode where every error was either
silently swallowed inside the pipeline (no retry at all) or burned all
3 retries on a guaranteed-to-fail parse.
"""

import asyncio

from celery.exceptions import Retry

from src.core.logger import get_logger
from src.ingestion.pipeline import IngestionResult
from src.services.storage_service import StorageConnectionError
from src.workers.celery_app import celery_app

log = get_logger(__name__)


# Errors worth retrying. Anything else either succeeded or is a
# permanent failure (handled inside the pipeline).
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    StorageConnectionError,
    ConnectionError,
    TimeoutError,
)


@celery_app.task(
    name="src.workers.tasks.ingest_document_task",
    bind=True,
    autoretry_for=_TRANSIENT_ERRORS,
    retry_backoff=True,  # exponential: 1s, 2s, 4s, 8s …
    retry_backoff_max=60,  # capped at 60s between retries
    retry_jitter=True,
    max_retries=3,
)
def ingest_document_task(self, document_id: str) -> dict:
    """
    Full ingestion pipeline for one document.

    Transient infra errors are retried automatically (via
    `autoretry_for`); terminal errors leave the document FAILED with a
    descriptive `error_message` that the UI can show. Anything else is
    re-raised so it lands in dead-letter / Sentry.
    """
    log.info(
        "task_ingest_started",
        document_id=document_id,
        attempt=self.request.retries + 1,
    )
    try:
        result = asyncio.run(_run_ingestion(document_id))
    except Retry:
        # Celery's own retry signal — don't wrap it
        raise
    except _TRANSIENT_ERRORS:
        # autoretry_for picks this up; just log
        log.warning(
            "task_ingest_transient_error",
            document_id=document_id,
            attempt=self.request.retries + 1,
        )
        raise
    except Exception as exc:
        # Unexpected — let it propagate so observability tools see it.
        # The pipeline catches everything it can and converts to FAILED;
        # if we get here, something escaped that contract.
        log.exception(
            "task_ingest_unexpected_error",
            document_id=document_id,
            error=str(exc),
        )
        raise

    log.info(
        "task_ingest_completed",
        document_id=document_id,
        chunks=result.chunks_created,
        status=result.status.value,
    )
    return {
        "document_id": document_id,
        "chunks": result.chunks_created,
        "status": result.status.value,
        "error": result.error,
    }


@celery_app.task(
    name="src.workers.tasks.delete_document_vectors_task",
    bind=True,
    autoretry_for=_TRANSIENT_ERRORS,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def delete_document_vectors_task(
    self, workspace_id: str, document_id: str, user_id: str
) -> dict:
    """Remove all Qdrant vectors for a deleted document."""
    log.info(
        "task_vector_delete_started",
        document_id=document_id,
        attempt=self.request.retries + 1,
    )
    try:
        asyncio.run(_run_vector_delete(workspace_id, document_id, user_id))
    except _TRANSIENT_ERRORS:
        log.warning(
            "task_vector_delete_transient_error",
            document_id=document_id,
        )
        raise
    return {"document_id": document_id, "deleted": True}


# ── Async Implementations ─────────────────────────────────────────


async def _run_ingestion(document_id: str) -> IngestionResult:
    """
    Async implementation of the ingestion pipeline.
    Creates its own DB session — Celery workers are separate processes.
    """
    from uuid import UUID
    from sqlalchemy import select

    from src.configs.config import get_settings
    from src.db.session import AsyncSessionLocal, engine
    from src.ingestion.pipeline import IngestionPipeline
    from src.models.document import Document, DocumentStatus
    from src.providers.embedding.factory import get_embedding_provider
    from src.services.storage_service import StorageService
    from src.vector_store.factory import get_vector_store

    # asyncio.run() closes the event loop when it finishes. asyncpg connections
    # in the pool are bound to that closed loop. On the next task's asyncio.run()
    # (new loop), pool_pre_ping tries to ping those stale connections and hits
    # "Event loop is closed". Disposing with close=False discards them without
    # trying to close them (which would also fail on the dead loop), forcing
    # fresh connections to be created on the current loop.
    engine.dispose(close=False)

    settings = get_settings()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.id == UUID(document_id))
        )
        document = result.scalar_one_or_none()
        if not document:
            log.warning("task_ingest_document_gone", document_id=document_id)
            return IngestionResult(
                document_id=document_id,
                chunks_created=0,
                status=DocumentStatus.FAILED,
                error="Document was deleted before ingestion could start.",
            )

        pipeline = IngestionPipeline(
            embedder=get_embedding_provider(),
            vector_store=get_vector_store(),
            storage=StorageService(),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        try:
            ingestion_result = await pipeline.run(document, db)
            await db.commit()
            return ingestion_result
        except Exception:
            # Pipeline raised (transient error path). Flush whatever
            # status we set before the failure so the UI keeps reflecting
            # the latest stage even while retries run.
            try:
                await db.commit()
            except Exception:
                await db.rollback()
            raise


async def _run_vector_delete(workspace_id: str, document_id: str, user_id: str) -> None:
    from src.db.session import engine
    from src.vector_store.factory import get_vector_store

    engine.dispose(
        close=False
    )  # clear stale asyncpg pool connections (see _run_ingestion)
    vs = get_vector_store()
    await vs.delete_document_vectors(workspace_id, document_id, user_id)
