"""
tests/unit/test_docs.py
───────────────────────
Keeps the prose documentation honest: every source path it names exists,
and it follows the project's writing rule of no em or en dashes.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(
    [
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("docs/*.html"),
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "deploy" / "README.md",
    ]
)
# `src/...` inside backticks or <code>, up to a "*" glob or a ":symbol" suffix.
SRC_PATH = re.compile(r"(?:`|<code>)(src/[\w./-]*)")
# U+2013 en dash and U+2014 em dash, built from code points so this file
# contains neither.
DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")


def test_docs_exist() -> None:
    assert len(DOCS) >= 6, DOCS


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_named_source_paths_exist(path: Path) -> None:
    missing = [
        ref
        for ref in SRC_PATH.findall(path.read_text(encoding="utf-8"))
        if not (ROOT / ref.rstrip("/")).exists()
    ]
    assert not missing, f"{path.name} names paths that do not exist: {missing}"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_or_en_dashes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = [n for n, line in enumerate(text.splitlines(), 1) if DASHES.search(line)]
    assert not lines, f"{path.name}: em or en dash on lines {lines}"


# A Celery task by its registered name, as the deploy guide's commands use it.
TASK_NAME = re.compile(r"\bsrc\.workers\.\w+\.\w+_task\b")


def test_tasks_named_in_the_deploy_guide_exist() -> None:
    """A renamed task must not leave a deploy command that calls nothing."""
    from src.workers.celery_app import celery_app

    celery_app.loader.import_default_modules()  # registers the included tasks
    guide = (ROOT / "deploy" / "README.md").read_text(encoding="utf-8")
    named = set(TASK_NAME.findall(guide))
    assert "src.workers.demo_tasks.refresh_sample_library_task" in named
    assert named <= set(celery_app.tasks), named - set(celery_app.tasks)
