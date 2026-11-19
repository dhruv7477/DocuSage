"""Unit tests for the small helpers behind rate limits and /health/deep."""

from types import SimpleNamespace

import pytest

from src.configs.config import get_settings
from src.core.rate_limit import client_ip_key, setting_limit
from src.core.readiness import redact_url

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("redis://:s3cret@redis:6379/0", "redis://redis:6379/0"),
        ("redis://user:s3cret@redis/1", "redis://redis/1"),
        ("http://qdrant:6333", "http://qdrant:6333"),
        ("redis://localhost:6379/0", "redis://localhost:6379/0"),
    ],
)
def test_redact_url_drops_credentials(url: str, expected: str) -> None:
    assert redact_url(url) == expected
    assert "s3cret" not in redact_url(url)


def test_setting_limit_reads_the_current_value(monkeypatch: pytest.MonkeyPatch) -> None:
    limit = setting_limit("rate_limit_auth")
    monkeypatch.setattr(get_settings(), "rate_limit_auth", "7/minute")
    assert limit() == "7/minute"


def test_client_ip_key_uses_the_client_address() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.9"), headers={})
    assert client_ip_key(request) == "ip:203.0.113.9"
