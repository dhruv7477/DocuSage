"""
tests/unit/test_coverage_config.py
──────────────────────────────────
.coveragerc measures every Python directory SonarQube Cloud analyses as
code (the tests are excluded there), so tested code is not reported as
uncovered.
"""

import configparser
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def _coverage_run() -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(REPO / ".coveragerc", encoding="utf-8")
    return parser["run"]


def test_measures_the_app_and_the_ci_tools() -> None:
    sources = {s.strip() for s in _coverage_run()["source"].split(",")}
    assert sources == {"src", "scripts"}


def test_reports_paths_relative_to_the_repository() -> None:
    assert _coverage_run().getboolean("relative_files")
