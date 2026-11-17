"""`.env.example` must load as a working configuration when copied to `.env`."""

from pathlib import Path

import pytest

from src.configs.config import Settings

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[2] / ".env.example"


def _template_keys() -> list[str]:
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    return [ln.split("=", 1)[0] for ln in lines if "=" in ln and not ln.startswith("#")]


def test_env_example_loads_as_a_valid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _template_keys():  # the template's values, not the test env's
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=TEMPLATE)

    assert settings.app_env == "development"
    assert settings.allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert len(settings.secret_key) >= 32
    assert settings.qdrant_collection_prefix == "docusage"


def test_env_example_has_no_empty_required_values() -> None:
    required = {"SECRET_KEY", "POSTGRES_PASSWORD", "ALLOWED_ORIGINS"}
    values = dict(
        ln.split("=", 1)
        for ln in TEMPLATE.read_text(encoding="utf-8").splitlines()
        if "=" in ln and not ln.startswith("#")
    )
    assert all(values.get(k) for k in required)
