"""
tests/unit/test_storage_settings.py
───────────────────────────────────
S3_* storage settings, and the old MINIO_* names that still work for one
release (DECISIONS D5).
"""

import pytest

from src.configs.config import DEPRECATED_STORAGE_NAMES, Settings
from src.core import deprecations
from src.services import storage_service

pytestmark = pytest.mark.unit

_REQUIRED = {
    "secret_key": "unit-test-secret-key-that-is-32-chars",
    "postgres_password": "x",
}


@pytest.fixture(autouse=True)
def _no_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test env may export either naming; these tests set their own.
    for old, new in DEPRECATED_STORAGE_NAMES.items():
        monkeypatch.delenv(old.upper(), raising=False)
        monkeypatch.delenv(new.upper(), raising=False)
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, **_REQUIRED, **values)


def test_defaults_use_the_s3_compatible_backend() -> None:
    settings = _settings()
    assert settings.storage_backend == "s3_compatible"
    assert settings.s3_endpoint == "localhost:9000"
    assert settings.deprecated_storage_settings() == []


def test_new_names_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S3_ENDPOINT", "seaweedfs:8333")
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECURE", "true")
    settings = _settings()
    assert (settings.s3_endpoint, settings.s3_access_key, settings.s3_secure) == (
        "seaweedfs:8333",
        "access",
        True,
    )
    assert settings.deprecated_storage_settings() == []


def test_old_names_still_work_and_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for old in DEPRECATED_STORAGE_NAMES:
        monkeypatch.setenv(old.upper(), "true" if old == "minio_secure" else old)
    monkeypatch.setenv("STORAGE_BACKEND", "minio")
    settings = _settings()

    assert settings.s3_endpoint == "minio_endpoint"
    assert settings.s3_access_key == "minio_access_key"
    assert settings.s3_secret_key == "minio_secret_key"
    assert settings.s3_bucket == "minio_bucket"
    assert settings.s3_secure is True
    assert settings.deprecated_storage_settings() == [
        "MINIO_ENDPOINT (use S3_ENDPOINT)",
        "MINIO_ACCESS_KEY (use S3_ACCESS_KEY)",
        "MINIO_SECRET_KEY (use S3_SECRET_KEY)",
        "MINIO_BUCKET (use S3_BUCKET)",
        "MINIO_SECURE (use S3_SECURE)",
        "STORAGE_BACKEND=minio (use s3_compatible)",
    ]


def test_new_name_wins_over_old_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_ENDPOINT", "old:9000")
    monkeypatch.setenv("S3_ENDPOINT", "new:8333")
    assert _settings().s3_endpoint == "new:8333"


@pytest.mark.parametrize(
    ("backend", "s3_compatible"),
    [("s3_compatible", True), ("minio", True), ("s3", False)],
)
def test_backend_selects_the_client(
    monkeypatch: pytest.MonkeyPatch, backend: str, s3_compatible: bool
) -> None:
    monkeypatch.setattr(storage_service.settings, "storage_backend", backend)
    assert storage_service._s3_compatible() is s3_compatible


def test_deprecated_settings_are_logged_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIO_ENDPOINT", "old:9000")
    warnings: list[dict] = []
    monkeypatch.setattr(
        deprecations.log, "warning", lambda event, **kw: warnings.append(kw)
    )

    used = deprecations.warn_deprecated_settings(_settings())

    assert used == ["MINIO_ENDPOINT (use S3_ENDPOINT)"]
    assert [w["setting"] for w in warnings] == used
