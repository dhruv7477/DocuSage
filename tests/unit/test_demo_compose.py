"""
tests/unit/test_demo_compose.py
───────────────────────────────
Static checks on docker/docker-compose.demo.yml and its example env file:
every app setting it passes is real, and the free-tier defaults hold.
"""

import re
from pathlib import Path

import pytest
import yaml

from src.configs.config import Settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.demo.yml"
ENV_EXAMPLE = ROOT / "docker" / ".env.demo.example"

# Read by uvicorn, glibc and the image entrypoint, not by Settings.
NOT_SETTINGS = {
    "FORWARDED_ALLOW_IPS",
    "WEB_CONCURRENCY",
    "MALLOC_ARENA_MAX",
    "SKIP_MIGRATIONS",
}
# Shut down by Groq on 2026-08-16.
RETIRED_GROQ_MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"}


def _app_env() -> dict[str, str]:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    return {**services["api"]["environment"], **services["worker"]["environment"]}


def _default(value: str) -> str:
    """The fallback in a "${NAME:-default}" value."""
    match = re.fullmatch(r"\$\{\w+:-([^}]*)\}", value)
    assert match, f"{value!r} has no ${{NAME:-default}} fallback"
    return match.group(1)


def test_every_app_variable_is_a_setting() -> None:
    unknown = {
        name
        for name in _app_env()
        if name not in NOT_SETTINGS and name.lower() not in Settings.model_fields
    }
    assert not unknown, f"not Settings fields (typo?): {sorted(unknown)}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("QUOTA_QUESTIONS_PER_DAY", "10"),
        ("DEMO_GLOBAL_QUESTIONS_PER_DAY", "50"),
        ("RATE_LIMIT_CHAT_PER_MINUTE", "10"),
        ("DOC_SUMMARY_ENABLED", "false"),
        ("LLM_REASONING_EFFORT", "low"),
        ("QUOTA_MAX_DOCUMENTS", "5"),
        ("QUOTA_MAX_UPLOAD_MB", "25"),
    ],
)
def test_free_tier_defaults_can_be_overridden(name: str, expected: str) -> None:
    # Each limit is a docker/.env setting with the free-tier value as the
    # fallback (DECISIONS D7).
    assert _default(_app_env()[name]) == expected


def test_example_env_names_current_models() -> None:
    values = dict(
        line.split("=", 1)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )
    models = {values["LLM_MODEL"], values["ROUTER_MODEL"], values["SUMMARY_MODEL"]}
    assert all(models)
    assert not models & RETIRED_GROQ_MODELS
