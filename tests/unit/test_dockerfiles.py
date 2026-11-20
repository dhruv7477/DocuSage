"""
tests/unit/test_dockerfiles.py
──────────────────────────────
Static checks on the two Dockerfiles: hardening that is easy to lose in an
edit or a dependency bump.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
DOCKER = REPO / "docker"
API = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
WEB = (DOCKER / "Dockerfile.web").read_text(encoding="utf-8")


def _final_user(dockerfile: str) -> str:
    users = re.findall(r"^USER\s+(\S+)", dockerfile, flags=re.MULTILINE)
    assert users, "no USER instruction"
    return users[-1]


def _context_sources(dockerfile: str) -> list[str]:
    """Build-context paths that COPY/ADD take (stage copies with --from= excluded)."""
    sources: list[str] = []
    for match in re.finditer(r"^(?:COPY|ADD)\s+(.+)$", dockerfile, flags=re.MULTILINE):
        words = match.group(1).split()
        if any(w.startswith("--from=") for w in words):
            continue
        args = [w for w in words if not w.startswith("--")]
        sources.extend(args[:-1])  # the last argument is the destination
    return sources


@pytest.mark.parametrize("dockerfile", [API, WEB], ids=["api", "web"])
def test_images_copy_named_paths_not_the_whole_context(dockerfile: str) -> None:
    sources = _context_sources(dockerfile)
    assert sources
    for source in sources:
        assert source.rstrip("/") not in {".", "./"}, "copies the whole build context"
        assert not set(source) & set("*?["), f"glob in COPY source {source!r}"
        assert (REPO / source).exists(), f"{source} is not in the repository"


def test_api_image_has_the_code_migrations_and_demo_samples_only() -> None:
    assert set(_context_sources(API)) == {
        "requirements.txt",
        "constraints.txt",
        "src",
        "alembic.ini",
        "demo",
        "docker/scripts/entrypoint.sh",
    }


@pytest.mark.parametrize("dockerfile", [API, WEB], ids=["api", "web"])
def test_images_run_as_an_unprivileged_user(dockerfile: str) -> None:
    assert _final_user(dockerfile).split(":")[0] not in {"root", "0"}


def test_api_installs_wheels_only_and_drops_pip() -> None:
    installs = [line for line in API.splitlines() if "pip install" in line]
    assert installs
    assert all("--only-binary :all:" in line for line in installs), installs
    assert "pip uninstall -y pip" in API


def test_web_installs_no_lifecycle_scripts() -> None:
    assert "npm ci --ignore-scripts" in WEB


def test_web_rebuilds_the_same_caddy_version_it_runs() -> None:
    # Dependabot bumps the two FROM lines; the xcaddy version must follow.
    builder = re.search(r"^FROM caddy:([\d.]+)-builder", WEB, flags=re.MULTILINE)
    runtime = re.search(r"^FROM caddy:([\d.]+)-alpine\s*$", WEB, flags=re.MULTILINE)
    built = re.search(r"xcaddy build v([\d.]+)", WEB)
    assert builder and runtime and built
    assert builder.group(1) == runtime.group(1) == built.group(1)
