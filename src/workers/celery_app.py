"""
src/workers/celery_app.py
──────────────────────────
Celery application factory.

Workers pick up tasks from the Redis broker.
Each task runs the ingestion pipeline in a background process,
keeping the FastAPI event loop free for HTTP requests.

Queue design:
    ingestion  — document processing (CPU + I/O heavy, concurrency=2)
    default    — lightweight tasks (cleanup, notifications)

Both queues are declared in `task_queues`, so a worker started without
-Q consumes both. Before, a plain `celery worker` listened only on the
implicit "celery" queue and uploads sat in "ingestion" forever.

Pre-warming
-----------
The first ingestion task in a fresh worker process used to spend several
seconds (or minutes on cold disk cache) loading the BGE embedder model
and Qdrant client — a tax paid before the user saw any progress.
`worker_process_init` warms the embedding singleton at worker startup
so that tax is paid exactly once, before any real document hits the
queue. Subsequent tasks reuse the cached singleton.
"""

from celery import Celery
from kombu import Queue
from celery.signals import worker_process_init, worker_ready

from src.configs.config import get_settings
from src.core.logger import get_logger

log = get_logger(__name__)

settings = get_settings()

celery_app = Celery(
    "privateai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_acks_late=True,  # Re-queue if worker crashes mid-task
    task_reject_on_worker_lost=True,  # Re-queue if worker process dies
    worker_prefetch_multiplier=1,  # One task at a time per worker (prevents OOM for large models)
    # Timeouts
    task_soft_time_limit=600,  # 10 min soft limit (sends SIGTERM)
    task_time_limit=660,  # 11 min hard limit (sends SIGKILL)
    # Result retention
    result_expires=86400,  # Keep results 24 hours
    # Silence Celery 5→6 migration warning about broker retry behaviour
    broker_connection_retry_on_startup=True,
    # Routing
    # Distinct routing keys, so a message bound for one queue never matches the other.
    task_queues=(
        Queue("ingestion", routing_key="ingestion"),
        Queue("default", routing_key="default"),
    ),
    task_default_queue="default",
    task_routes={
        "src.workers.tasks.ingest_document_task": {"queue": "ingestion"},
        "src.workers.tasks.delete_document_vectors_task": {"queue": "default"},
        "src.workers.tasks.fail_stale_documents_task": {"queue": "default"},
    },
    # Periodic jobs (needs `celery beat`, or a worker started with -B)
    beat_schedule={
        "fail-stale-documents": {
            "task": "src.workers.tasks.fail_stale_documents_task",
            "schedule": 600.0,  # every 10 minutes
        },
    },
)


@worker_process_init.connect
def _prewarm_worker_models(**_kwargs) -> None:
    """
    Load the embedder (and verify Qdrant connectivity) once per worker
    process so the first real ingestion task doesn't pay the cold-start
    tax. Failure here is logged but doesn't abort the worker — the task
    itself will retry transient errors via `autoretry_for`.
    """
    try:
        from src.providers.embedding.factory import get_embedding_provider

        from src.workers import runtime

        embedder = get_embedding_provider()
        log.info("worker_prewarm_start", provider=embedder.provider_name)
        ok = runtime.run(embedder.health())
        log.info(
            "worker_prewarm_done",
            provider=embedder.provider_name,
            ok=ok,
            dim=embedder.dimension,
        )
    except Exception as exc:
        log.warning("worker_prewarm_failed", error=str(exc))


@worker_ready.connect
def _sweep_stale_documents_on_start(**_kwargs) -> None:
    """Fail documents a previous worker left in flight (crash, kill, deploy)."""
    try:
        from src.workers.tasks import fail_stale_documents_task

        fail_stale_documents_task.delay()
    except Exception as exc:
        log.warning("stale_sweep_dispatch_failed", error=str(exc))
