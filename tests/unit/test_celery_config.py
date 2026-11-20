"""
tests/unit/test_celery_config.py
────────────────────────────────
Worker reliability settings, checked through Celery's own failure
handling rather than by reading the config dict.
"""

from unittest.mock import Mock

import pytest
from billiard.einfo import ExceptionInfo
from billiard.exceptions import WorkerLostError
from celery.worker.request import Request

from src.workers.celery_app import celery_app
from src.workers.tasks import ingest_document_task

pytestmark = pytest.mark.unit


def _request_for(task, on_ack: Mock, on_reject: Mock) -> Request:
    msg = celery_app.amqp.as_task_v2("task-1", task.name, args=("doc-1",))
    message = Mock(
        headers=msg.headers,
        body=msg.body,
        delivery_info={},
        properties={},
        content_type="application/json",
        content_encoding="utf-8",
    )
    return Request(
        message,
        app=celery_app,
        task=task,
        on_ack=on_ack,
        on_reject=on_reject,
        decoded=True,
        body=msg.body,
        headers=msg.headers,
    )


def _worker_lost() -> ExceptionInfo:
    try:
        raise WorkerLostError("Worker exited prematurely: signal 9 (SIGKILL).")
    except WorkerLostError:
        return ExceptionInfo()


def test_ingestion_acks_late() -> None:
    assert ingest_document_task.acks_late is True


def test_lost_worker_fails_the_task_instead_of_redelivering(monkeypatch) -> None:
    # A file that kills the pool process must not be redelivered forever
    # (DECISIONS D4): the message is acked and the task is marked failed.
    backend = Mock()
    # Patch the attribute behind the `backend` property, which has no deleter.
    monkeypatch.setattr(ingest_document_task, "_backend", backend)
    on_ack, on_reject = Mock(), Mock()

    _request_for(ingest_document_task, on_ack, on_reject).on_failure(_worker_lost())

    on_ack.assert_called_once()
    on_reject.assert_not_called()
    backend.mark_as_failure.assert_called_once()
    assert isinstance(backend.mark_as_failure.call_args.args[1], WorkerLostError)


def test_a_new_pool_process_has_time_to_load_the_embedder() -> None:
    # Each new pool process imports torch and loads the embedder before it
    # reports up. Celery's 4 s default killed every replacement in a loop.
    assert celery_app.conf.worker_proc_alive_timeout >= 60
