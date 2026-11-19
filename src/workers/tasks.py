"""
src/workers/tasks.py
──────────────────────
Celery task definitions.

Tasks are synchronous and the pipeline is async. Every task runs its
coroutine on the worker process's long-lived event loop
(`src.workers.runtime.run`), never with asyncio.run(), so pooled
database and Qdrant connections survive from one task to the next.

Retry policy
------------
Only transient errors trigger retries. Terminal errors (parse failure,
empty document, etc.) flow back as a FAILED `IngestionResult` without
an exception, which Celery treats as success and so won't retry.

Failure policy
--------------
A document must never stay in an in-flight status forever:
- `IngestionTask.on_failure` marks it FAILED when the task fails for
  good (retries exhausted, or an error the pipeline did not handle).
- `fail_stale_documents_task` marks it FAILED when it has not moved for
  `ingestion_stale_after_minutes` (worker killed, hard time limit, lost
  message). It runs from Celery beat and once when a worker starts.
"""

from datetime import timedelta
from typing import Any
from uuid import UUID

from celery import Task
from celery.exceptions import Retry
from sqlalchemy import func, select, update

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.ingestion.pipeline import IngestionResult
from src.models.document import DOCUMENT_IN_FLIGHT_STATUSES, Document, DocumentStatus
from src.services.storage_service import StorageConnectionError
from src.workers import runtime
from src.workers.celery_app import celery_app

log = get_logger(__name__)


# Errors worth retrying. Anything else either succeeded or is a
# permanent failure (handled inside the pipeline).
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    StorageConnectionError,
    ConnectionError,
    TimeoutError,
)

# Sorted so the generated SQL (and its statement cache key) is stable.
_IN_FLIGHT = sorted(DOCUMENT_IN_FLIGHT_STATUSES)

FAILED_AFTER_RETRIES_MESSAGE = (
    "Processing failed after several attempts. Please upload the file again."
)
STALE_MESSAGE = "Processing did not finish in time. Please upload the file again."


# ── Document status helpers ───────────────────────────────────────


async def mark_document_failed(document_id: str, message: str) -> bool:
    """Mark one in-flight document FAILED. Returns False if it was not in flight."""
    async with runtime.session_factory()() as db:
        result = await db.execute(
            update(Document)
            .where(
                Document.id == UUID(document_id),
                Document.status.in_(_IN_FLIGHT),
            )
            .values(status=DocumentStatus.FAILED, error_message=message, progress=0)
        )
        await db.commit()
        return result.rowcount > 0


async def fail_stale_documents(older_than: timedelta) -> int:
    """Mark every in-flight document untouched for `older_than` FAILED.

    Compares against the database clock so app and DB clock skew can't
    fail a document early. Returns the number of documents changed.
    """
    async with runtime.session_factory()() as db:
        result = await db.execute(
            update(Document)
            .where(
                Document.status.in_(_IN_FLIGHT),
                Document.updated_at < func.now() - older_than,
            )
            .values(
                status=DocumentStatus.FAILED, error_message=STALE_MESSAGE, progress=0
            )
        )
        await db.commit()
        return result.rowcount


class IngestionTask(Task):
    """Task base that marks the document FAILED when the task fails for good."""

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        document_id = args[0] if args else kwargs.get("document_id")
        if document_id is None:
            log.error("task_ingest_failed_without_document_id", task_id=task_id)
            return
        try:
            marked = runtime.run(
                mark_document_failed(str(document_id), FAILED_AFTER_RETRIES_MESSAGE)
            )
        except Exception as mark_exc:
            log.exception(
                "task_ingest_mark_failed_error",
                document_id=str(document_id),
                error=str(mark_exc),
            )
            return
        log.error(
            "task_ingest_failed",
            document_id=str(document_id),
            error_type=type(exc).__name__,
            marked_failed=marked,
        )


# ── Tasks ─────────────────────────────────────────────────────────


@celery_app.task(
    name="src.workers.tasks.ingest_document_task",
    bind=True,
    base=IngestionTask,
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
    re-raised, and `IngestionTask.on_failure` marks the document FAILED.
    """
    log.info(
        "task_ingest_started",
        document_id=document_id,
        attempt=self.request.retries + 1,
    )
    try:
        result = runtime.run(_run_ingestion(document_id))
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
        runtime.run(_run_vector_delete(workspace_id, document_id, user_id))
    except _TRANSIENT_ERRORS:
        log.warning(
            "task_vector_delete_transient_error",
            document_id=document_id,
        )
        raise
    return {"document_id": document_id, "deleted": True}


@celery_app.task(name="src.workers.tasks.fail_stale_documents_task")
def fail_stale_documents_task() -> int:
    """Fail documents stuck in an in-flight status (see module docstring)."""
    minutes = get_settings().ingestion_stale_after_minutes
    count = runtime.run(fail_stale_documents(timedelta(minutes=minutes)))
    if count:
        log.warning("stale_documents_failed", count=count, older_than_min=minutes)
    return count


# ── Async Implementations ─────────────────────────────────────────


async def _run_ingestion(document_id: str) -> IngestionResult:
    """Run the ingestion pipeline for one document in its own DB session."""

    from src.ingestion.pipeline import IngestionPipeline
    from src.providers.embedding.factory import get_embedding_provider
    from src.services.storage_service import StorageService
    from src.vector_store.factory import get_vector_store

    settings = get_settings()

    async with runtime.session_factory()() as db:
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
            # Pipeline raised (transient error path). Commit whatever
            # status we set before the failure so the UI keeps reflecting
            # the latest stage even while retries run.
            try:
                await db.commit()
            except Exception:
                await db.rollback()
            raise


async def _run_vector_delete(workspace_id: str, document_id: str, user_id: str) -> None:
    from src.vector_store.factory import get_vector_store

    vs = get_vector_store()
    await vs.delete_document_vectors(workspace_id, document_id, user_id)
