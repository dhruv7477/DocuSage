"""
tests/unit/test_sonar_config.py
───────────────────────────────
Static checks on sonar-project.properties, the analysis settings the
"SonarQube Cloud" CI job uses.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROPERTIES = Path(__file__).resolve().parents[2] / "sonar-project.properties"

# SonarQube Cloud's analysis scope covers the whole repository and warns
# that these keys "have no effect".
IGNORED_SCOPE_KEYS = {"sonar.sources", "sonar.tests", "sonar.test.inclusions"}


def _properties() -> dict[str, str]:
    """Parse the key=value lines of the properties file (no line continuations)."""
    props: dict[str, str] = {}
    for raw in PROPERTIES.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep, f"not a key=value line: {raw!r}"
        props[key.strip()] = value.strip()
    return props


def _patterns(key: str) -> list[str]:
    return [p.strip() for p in _properties()[key].split(",")]


def test_sets_no_scope_keys_the_analysis_ignores() -> None:
    assert not IGNORED_SCOPE_KEYS & _properties().keys()


def test_python_tests_are_not_counted_as_code_to_cover() -> None:
    assert "tests/**" in _patterns("sonar.coverage.exclusions")


def test_frontend_tests_stay_out_of_the_analysis() -> None:
    assert {"src/**/__tests__/**", "src/test/**"} <= set(_patterns("sonar.exclusions"))
