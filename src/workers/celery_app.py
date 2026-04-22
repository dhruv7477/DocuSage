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
"""

from celery import Celery

from src.configs.config import get_settings

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
    task_acks_late=True,              # Re-queue if worker crashes mid-task
    task_reject_on_worker_lost=True,  # Re-queue if worker process dies
    worker_prefetch_multiplier=1,     # One task at a time per worker (prevents OOM for large models)

    # Timeouts
    task_soft_time_limit=600,         # 10 min soft limit (sends SIGTERM)
    task_time_limit=660,              # 11 min hard limit (sends SIGKILL)

    # Result retention
    result_expires=86400,             # Keep results 24 hours

    # Routing
    task_routes={
        "src.workers.tasks.ingest_document_task": {"queue": "ingestion"},
        "src.workers.tasks.delete_document_vectors_task": {"queue": "default"},
    },
)