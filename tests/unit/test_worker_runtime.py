"""Unit tests for the per-process worker event loop and task failure handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.workers import runtime
from src.workers.tasks import FAILED_AFTER_RETRIES_MESSAGE, ingest_document_task

pytestmark = pytest.mark.unit


async def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_run_reuses_one_loop_across_calls() -> None:
    assert runtime.run(_current_loop()) is runtime.run(_current_loop())


def test_loop_bound_objects_survive_between_runs() -> None:
    """A future made in one task can be awaited in the next (asyncio.run fails here)."""

    async def make_future() -> asyncio.Future[str]:
        return asyncio.get_running_loop().create_future()

    async def resolve_and_await(fut: asyncio.Future[str]) -> str:
        fut.set_result("ok")
        return await fut

    fut = runtime.run(make_future())
    assert runtime.run(resolve_and_await(fut)) == "ok"


def test_new_process_gets_a_new_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    parent_loop = runtime.run(_current_loop())
    monkeypatch.setattr(runtime.os, "getpid", lambda: -1)  # pretend we forked

    child_loop = runtime.run(_current_loop())

    assert child_loop is not parent_loop


def test_on_failure_marks_the_document_failed() -> None:
    with patch(
        "src.workers.tasks.mark_document_failed", new=AsyncMock(return_value=True)
    ) as mark:
        ingest_document_task.on_failure(
            RuntimeError("boom"), "task-1", ("doc-1",), {}, None
        )

    mark.assert_awaited_once_with("doc-1", FAILED_AFTER_RETRIES_MESSAGE)


def test_on_failure_logs_and_returns_when_marking_fails() -> None:
    with (
        patch(
            "src.workers.tasks.mark_document_failed",
            new=AsyncMock(side_effect=ConnectionError("db down")),
        ),
        patch("src.workers.tasks.log") as log,
    ):
        ingest_document_task.on_failure(
            RuntimeError("boom"), "task-1", ("doc-1",), {}, None
        )

    log.exception.assert_called_once()


def test_worker_without_queue_flag_consumes_ingestion_and_default() -> None:
    from src.workers.celery_app import celery_app

    assert set(celery_app.amqp.queues) >= {"ingestion", "default"}
    route = celery_app.amqp.router.route({}, "src.workers.tasks.ingest_document_task")
    assert route["queue"].name == "ingestion"


def test_each_queue_has_its_own_routing_key() -> None:
    from src.workers.celery_app import celery_app

    keys = {name: q.routing_key for name, q in celery_app.amqp.queues.items()}
    assert keys["ingestion"] == "ingestion"
    assert keys["default"] == "default"
