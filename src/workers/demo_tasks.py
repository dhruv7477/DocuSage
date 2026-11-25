"""
src/workers/demo_tasks.py
─────────────────────────
Celery tasks for the public demo (DEMO_MODE=true). Each one returns at once
when demo mode is off, so the beat schedule is harmless elsewhere.
See src/demo/samples.py for what they do.
"""

import uuid

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.demo.samples import (
    SeedOutcome,
    ensure_library,
    load_library,
    refresh_sample_copies,
    seed_user,
    wipe_user_uploads,
)
from src.models.document import DocumentStatus
from src.workers import runtime
from src.workers.celery_app import celery_app

log = get_logger(__name__)

# A new account's seeding waits for its own row to be committed and, on a
# fresh install, for the library to finish ingesting: 40 tries, 15 s apart.
SEED_RETRY_SECONDS = 15
SEED_MAX_RETRIES = 40


def queue_sample_documents(user_id: uuid.UUID) -> None:
    """Queue the sample copies for a new account. Never fails the caller."""
    try:
        seed_sample_documents_task.delay(str(user_id))
    except Exception as exc:
        log.warning("sample_seed_dispatch_failed", user_id=str(user_id), error=str(exc))


@celery_app.task(name="src.workers.demo_tasks.ensure_sample_library_task")
def ensure_sample_library_task() -> int:
    """Register and ingest any sample file the library does not have yet."""
    if not get_settings().demo_mode:
        return 0
    from src.workers.tasks import ingest_document_task

    to_ingest = runtime.run(_ensure_library())
    for document_id in to_ingest:
        ingest_document_task.delay(document_id)
    return len(to_ingest)


@celery_app.task(
    bind=True,
    name="src.workers.demo_tasks.seed_sample_documents_task",
    max_retries=SEED_MAX_RETRIES,
)
def seed_sample_documents_task(self, user_id: str) -> str:
    """Give one account its own copies of the sample documents."""
    if not get_settings().demo_mode:
        return "disabled"
    outcome = runtime.run(_seed(uuid.UUID(user_id)))
    if outcome in (SeedOutcome.LIBRARY_NOT_READY, SeedOutcome.USER_NOT_FOUND):
        log.info("sample_seed_waiting", user_id=user_id, reason=outcome.value)
        raise self.retry(countdown=SEED_RETRY_SECONDS)
    return outcome.value


@celery_app.task(name="src.workers.demo_tasks.wipe_user_uploads_task")
def wipe_user_uploads_task() -> int:
    """Delete every uploaded document, keeping the samples."""
    if not get_settings().demo_mode:
        return 0
    return runtime.run(_wipe())


@celery_app.task(
    name="src.workers.demo_tasks.refresh_sample_library_task",
    # Re-ingests every sample in one run; allow for a slow 2 GB server.
    soft_time_limit=1800,
    time_limit=1860,
)
def refresh_sample_library_task() -> dict[str, int]:
    """
    Re-ingest the sample library with the running code, then copy it again
    into every account's sample copies. Run once after a deploy that changes
    how documents are chunked (deploy/README.md, "Update to a new version").
    Uploads are left to the nightly wipe. If a sample fails to ingest, the
    copies are not touched.
    """
    if not get_settings().demo_mode:
        return {"library": 0, "copies": 0}
    from src.workers.tasks import _run_ingestion

    document_ids = runtime.run(_library_document_ids())
    for document_id in document_ids:
        result = runtime.run(_run_ingestion(document_id))
        if result.status != DocumentStatus.COMPLETED:
            log.error(
                "sample_library_refresh_failed",
                document_id=document_id,
                error=result.error,
            )
            raise RuntimeError(f"sample {document_id} failed to ingest: {result.error}")
    copies = runtime.run(_refresh_copies())
    log.info("sample_library_refreshed", documents=len(document_ids), copies=copies)
    return {"library": len(document_ids), "copies": copies}


async def _library_document_ids() -> list[str]:
    async with runtime.session_factory()() as db:
        return [str(d.id) for d in (await load_library(db)).documents]


async def _refresh_copies() -> int:
    from src.vector_store.factory import get_vector_store

    async with runtime.session_factory()() as db:
        return await refresh_sample_copies(db, get_vector_store())


async def _ensure_library() -> list[str]:
    from src.services.storage_service import StorageService

    async with runtime.session_factory()() as db:
        to_ingest = await ensure_library(db, StorageService())
        await db.commit()
        return to_ingest


async def _seed(user_id: uuid.UUID) -> SeedOutcome:
    from src.vector_store.factory import get_vector_store

    async with runtime.session_factory()() as db:
        return await seed_user(db, get_vector_store(), user_id)


async def _wipe() -> int:
    from src.services.storage_service import StorageService
    from src.vector_store.factory import get_vector_store

    async with runtime.session_factory()() as db:
        return await wipe_user_uploads(db, StorageService(), get_vector_store())
