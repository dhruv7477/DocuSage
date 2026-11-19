"""
tests/unit/test_workflows.py
────────────────────────────
Static checks on the GitHub Actions workflows.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
    ]


def test_workflows_exist() -> None:
    assert WORKFLOWS, f"no workflows found in {WORKFLOW_DIR}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_publishes_images(path: Path) -> None:
    # Images are built from the clone on the VM; nothing is pushed to a
    # registry (DECISIONS D1).
    workflow = _load(path)
    for step in _steps(workflow):
        uses = step.get("uses", "")
        assert not uses.startswith("docker/login-action"), step
        assert step.get("with", {}).get("push") is not True, step
    scopes = [workflow.get("permissions") or {}] + [
        job.get("permissions") or {} for job in workflow.get("jobs", {}).values()
    ]
    for scope in scopes:
        if isinstance(scope, dict):
            assert scope.get("packages") != "write", path.name
