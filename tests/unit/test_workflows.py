"""
tests/unit/test_workflows.py
────────────────────────────
Static checks on the GitHub Actions workflows.
"""

import re
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


_PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def _jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return workflow.get("jobs", {})


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    for step in _steps(_load(path)):
        uses = step.get("uses")
        if uses and not uses.startswith("./"):
            assert _PINNED_ACTION.match(uses), f"{path.name}: {uses} is not a SHA"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_default_permissions_are_read_only(path: Path) -> None:
    permissions = _load(path).get("permissions")
    assert isinstance(permissions, dict), f"{path.name}: no top-level permissions"
    assert "write" not in permissions.values(), path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_checkout_does_not_keep_the_token(path: Path) -> None:
    for step in _steps(_load(path)):
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_container_images_in_env_are_pinned_by_digest(path: Path) -> None:
    for name, value in (_load(path).get("env") or {}).items():
        if name.endswith("_IMAGE") and name != "DOCKER_IMAGE":
            assert "@sha256:" in str(value), f"{path.name}: {name} has no digest"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_pip_installs_wheels_only(path: Path) -> None:
    for step in _steps(_load(path)):
        for line in str(step.get("run", "")).splitlines():
            if "pip install" in line:
                assert "--only-binary :all:" in line, f"{path.name}: {line.strip()}"
