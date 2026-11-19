"""
tests/integration/test_celery_broker.py
───────────────────────────────────────
The worker can open a channel on the Redis broker with the installed
kombu and redis-py versions.

Regression: a fresh install resolved kombu 5.7.0, which raises
VersionMismatch ("Redis transport requires redis-py versions 6.1.0 or
later") the moment a channel opens, so no upload could be queued and
the worker exited at start.
"""

import socket
from urllib.parse import urlparse

import pytest

from src.configs.config import get_settings
from src.workers.celery_app import celery_app


def _broker_reachable() -> bool:
    url = urlparse(get_settings().celery_broker_url)
    try:
        with socket.create_connection(
            (url.hostname or "localhost", url.port or 6379), timeout=2
        ):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _broker_reachable(), reason="needs a live Redis broker"),
]


def test_worker_can_open_a_broker_channel() -> None:
    with celery_app.connection_for_write() as conn:
        conn.ensure_connection(max_retries=1)
        channel = conn.default_channel  # kombu checks the redis-py version here
        assert channel is not None
