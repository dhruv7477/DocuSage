"""
tests/unit/test_shell_scripts.py
────────────────────────────────
Bash scripts use the [[ ]] conditional, which does no word splitting or
globbing. POSIX sh scripts (the container entrypoints) keep [ ], since sh
has no [[.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = ("deploy", "docker", "scripts", "tests")

# A "[" test command where a command can start: line start, after ; & | (
# or after a keyword such as if, elif, while, until, then, do, !.
SINGLE_BRACKET_TEST = re.compile(
    r"(?:^|[;&|(]|\b(?:if|elif|while|until|then|do)\b|!)\s*\[\s"
)


def _bash_scripts() -> list[Path]:
    scripts = (p for d in SCRIPT_DIRS for p in sorted((REPO / d).rglob("*.sh")))
    return [
        p for p in scripts if "bash" in p.read_text(encoding="utf-8").split("\n", 1)[0]
    ]


def test_there_are_bash_scripts_to_check() -> None:
    names = {p.name for p in _bash_scripts()}
    assert {"setup-vm.sh", "backup-db.sh"} <= names


@pytest.mark.parametrize("script", _bash_scripts(), ids=lambda p: p.name)
def test_bash_scripts_use_double_bracket_conditionals(script: Path) -> None:
    offending = [
        f"{script.name}:{number}: {line.strip()}"
        for number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), 1
        )
        if not line.lstrip().startswith("#") and SINGLE_BRACKET_TEST.search(line)
    ]
    assert not offending, offending


@pytest.mark.parametrize(
    "line",
    [
        'if [ -n "$x" ]; then',
        'while [ "$i" -lt 3 ]; do',
        "[ -f x ] && echo y",
        "a && [ -z b ]",
    ],
)
def test_the_check_finds_single_bracket_tests(line: str) -> None:
    assert SINGLE_BRACKET_TEST.search(line)


@pytest.mark.parametrize(
    "line", ['if [[ -n "$x" ]]; then', 'arr=("${spec[@]}")', 'echo "[ok]"', "x=${a[0]}"]
)
def test_the_check_ignores_other_brackets(line: str) -> None:
    assert not SINGLE_BRACKET_TEST.search(line)
