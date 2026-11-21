"""
tests/unit/test_demo_memory.py
──────────────────────────────
The 2 GB profile of docker/docker-compose.demo.yml (an AWS t4g.small has
2 GB of RAM and a 4 GB swap file from deploy/setup-vm.sh): every service
has a memory limit, the limits add up to less than the budget, only the two
Python containers may page out (a bounded amount), and the processes are
set up to stay inside their limits. The "Demo stack end to end" CI job runs
the stack with these limits and fails on any OOM kill.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.demo.yml"
# Leaves about 400 MB of the 2 GB for the kernel, dockerd, containerd and sshd.
BUDGET_MIB = 1600
# torch and sentence-transformers import about 150 MB each that is never
# used again, which the API and the worker may page out. Qdrant may page out
# the collections nobody is searching. Nothing else swaps.
SWAP_ALLOWED = {"api", "worker", "qdrant"}
MAX_SWAP_MIB = 512
# Connections the app can open: the API's pool (10 + 20 overflow) and the
# worker's (2 + 2), plus a few for psql and the nightly backup.
DB_CONNECTIONS_NEEDED = 10 + 20 + 2 + 2 + 3

SERVICES = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]

_UNITS = {"k": 1 / 1024, "m": 1, "g": 1024}


def _mib(value: str) -> float:
    match = re.fullmatch(r"(\d+)([kmg])", str(value).lower())
    assert match, f"unexpected memory size {value!r}"
    return int(match.group(1)) * _UNITS[match.group(2)]


def _postgres_settings() -> dict[str, str]:
    command = SERVICES["postgres"]["command"]
    pairs = [command[i + 1] for i, arg in enumerate(command) if arg == "-c"]
    return dict(p.split("=", 1) for p in pairs)


def _swap_mib(config: dict) -> float:
    return _mib(config["memswap_limit"]) - _mib(config["mem_limit"])


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_every_service_has_a_memory_limit(service: str) -> None:
    config = SERVICES[service]
    assert "mem_limit" in config, f"{service} has no mem_limit"
    swap = _swap_mib(config)
    if service in SWAP_ALLOWED:
        assert 0 < swap <= MAX_SWAP_MIB
    else:
        # A leak shows up as an OOM kill (and a restart), which CI catches,
        # instead of a slowly thrashing server.
        assert swap == 0


def test_the_limits_fit_the_budget() -> None:
    total = sum(_mib(s["mem_limit"]) for s in SERVICES.values())
    assert total <= BUDGET_MIB, f"{total:.0f} MiB of limits, budget {BUDGET_MIB}"


def test_the_swap_allowed_fits_the_swap_file() -> None:
    # Well inside the 4 GB swap file, which also serves image builds and the host.
    assert sum(_swap_mib(s) for s in SERVICES.values()) <= 1536


def test_qdrant_collections_start_small() -> None:
    env = SERVICES["qdrant"]["environment"]
    assert env["QDRANT__STORAGE__WAL__WAL_CAPACITY_MB"] == "1"
    assert env["QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER"] == "1"


def test_python_containers_limit_malloc_arenas() -> None:
    for service in ("api", "worker"):
        assert SERVICES[service]["environment"]["MALLOC_ARENA_MAX"] == "2"


def test_one_api_process() -> None:
    assert SERVICES["api"]["environment"]["WEB_CONCURRENCY"] == "1"


def test_one_worker_process_that_is_recycled() -> None:
    command = SERVICES["worker"]["command"]
    assert "--concurrency=1" in command
    match = re.search(r"--max-tasks-per-child=(\d+)", command)
    assert match and int(match.group(1)) > 0


def test_models_run_in_small_batches() -> None:
    env = SERVICES["api"]["environment"]
    assert int(env["EMBEDDING_BATCH_SIZE"]) <= 8
    assert int(env["RERANKER_BATCH_SIZE"]) <= 8


def test_postgres_is_sized_for_the_limit() -> None:
    settings = _postgres_settings()
    shared_buffers = _mib(settings["shared_buffers"].lower().removesuffix("b"))
    assert shared_buffers <= _mib(SERVICES["postgres"]["mem_limit"]) / 4
    assert int(settings["max_connections"]) >= DB_CONNECTIONS_NEEDED
    assert settings["jit"] == "off"


def test_redis_never_evicts_queued_tasks() -> None:
    command = " ".join(SERVICES["redis"]["command"])
    maxmemory = re.search(r"--maxmemory (\d+[kmg])b?\b", command)
    assert maxmemory, "redis has no --maxmemory"
    assert _mib(maxmemory.group(1)) < _mib(SERVICES["redis"]["mem_limit"])
    # The Celery broker lives in Redis: evicting keys would drop tasks.
    assert "--maxmemory-policy noeviction" in command


@pytest.mark.parametrize("service", ["caddy", "seaweedfs"])
def test_go_services_collect_garbage_before_their_limit(service: str) -> None:
    limit = _mib(SERVICES[service]["mem_limit"])
    gomemlimit = SERVICES[service]["environment"]["GOMEMLIMIT"]
    assert gomemlimit.endswith("MiB")
    assert int(gomemlimit.removesuffix("MiB")) < limit
