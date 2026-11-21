"""
Memory watch and report for the demo stack's end-to-end run in CI.

    python3 tests/e2e/memory_report.py watch DIR           # in the background
    python3 tests/e2e/memory_report.py report DIR [--project NAME]

`watch` records Docker's OOM events and samples `docker stats` every few
seconds into DIR until it is stopped (SIGTERM or Ctrl+C). `report` then
prints one row per container of the compose project: its memory limit and
swap allowance, the highest sampled usage, the kernel's peaks for the
container's cgroup (memory and swap) and the number of OOM kills. The table also goes to $GITHUB_STEP_SUMMARY when that is
set. `report` exits 1 if any process in any container was OOM-killed.

Standard library only: it runs on the runner's own python3, before any
setup-python step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

STATS_FILE = "stats.jsonl"
EVENTS_FILE = "oom-events.jsonl"
SAMPLE_SECONDS = 3.0
DEFAULT_PROJECT = "docusage-demo"

_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}
_SIZE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)\s*$")


@dataclass(frozen=True)
class ContainerMemory:
    """What the report knows about one container."""

    name: str
    service: str
    limit: int  # bytes; 0 means no limit
    swap_limit: int  # bytes of swap allowed on top of `limit`
    sampled_peak: int | None  # bytes, from docker stats (page cache excluded)
    cgroup_peak: int | None  # bytes, memory.peak (page cache included)
    swap_peak: int | None  # bytes, memory.swap.peak
    oom_kills: int  # from memory.events, Docker's OOM events and State.OOMKilled
    restarts: int


def parse_size(text: str) -> int:
    """'512.3MiB' -> bytes. Accepts docker stats' binary and decimal units."""
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r}")
    number, unit = match.groups()
    try:
        return round(float(number) * _UNITS[unit.lower()])
    except KeyError:
        raise ValueError(f"unknown unit in {text!r}") from None


def peaks_from_stats(lines: list[str]) -> dict[str, int]:
    """Highest MemUsage per container name in `docker stats --format json` lines.

    A container that is starting or stopping shows "-- / --"; those samples
    are skipped.
    """
    peaks: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        usage = row["MemUsage"].split("/")[0].strip()
        if usage.startswith("--"):
            continue
        used = parse_size(usage)
        name = row["Name"]
        peaks[name] = max(used, peaks.get(name, 0))
    return peaks


def oom_counts_from_events(lines: list[str]) -> dict[str, int]:
    """OOM events per container name in `docker events --format json` lines."""
    counts: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("Action") != "oom":
            continue
        name = event.get("Actor", {}).get("Attributes", {}).get("name", "?")
        counts[name] = counts.get(name, 0) + 1
    return counts


def oom_kills_from_memory_events(text: str) -> int:
    """The oom_kill counter in a cgroup v2 memory.events file."""
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill":
            return int(value)
    return 0


def _mib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024**2:.0f} MiB"


def markdown_table(rows: list[ContainerMemory]) -> str:
    """The job-summary table, with a total of the limits."""
    lines = [
        "### Demo stack memory",
        "",
        "| Service | Limit | Swap allowed | Peak (docker stats) "
        "| Peak (cgroup, with page cache) | Peak swap | OOM kills | Restarts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r.service):
        limit = _mib(r.limit) if r.limit else "none"
        lines.append(
            f"| {r.service} | {limit} | {_mib(r.swap_limit)} | {_mib(r.sampled_peak)} "
            f"| {_mib(r.cgroup_peak)} | {_mib(r.swap_peak)} | {r.oom_kills} "
            f"| {r.restarts} |"
        )
    total = sum(r.limit for r in rows)
    swap = sum(r.swap_limit for r in rows)
    sampled = sum(r.sampled_peak or 0 for r in rows)
    lines.append(
        f"| **total** | **{_mib(total)}** | **{_mib(swap)}** | **{_mib(sampled)}** "
        "| | | | |"
    )
    lines += [
        "",
        "Docker stats leaves out reclaimable page cache and is sampled every "
        f"{SAMPLE_SECONDS:.0f} s; the cgroup peak is the kernel's exact high-water "
        "mark, page cache included.",
    ]
    return "\n".join(lines) + "\n"


def swap_allowance(host_config: dict) -> int:
    """Swap a container may use, from `docker inspect`'s HostConfig."""
    memory = host_config.get("Memory") or 0
    total = host_config.get("MemorySwap") or 0
    if memory and total > 0:
        return max(total - memory, 0)
    return 0  # no limit set, or no swap accounting


def _int_file(text: str | None) -> int | None:
    return int(text) if text and text.strip().isdigit() else None


def oom_killed(rows: list[ContainerMemory]) -> list[str]:
    """Services with at least one OOM kill."""
    return sorted(r.service for r in rows if r.oom_kills)


# ── Docker access ─────────────────────────────────────────────────


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout


def _read_in_container(name: str, path: str) -> str | None:
    """A cgroup file as the container sees it (its own cgroup namespace)."""
    result = subprocess.run(
        ["docker", "exec", name, "cat", path], capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def collect(project: str, out_dir: Path) -> list[ContainerMemory]:
    """One ContainerMemory per container of the compose project."""
    ids = _docker(
        "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"
    ).split()
    if not ids:
        raise SystemExit(f"no containers for compose project {project!r}")
    stats_path = out_dir / STATS_FILE
    events_path = out_dir / EVENTS_FILE
    sampled = peaks_from_stats(
        stats_path.read_text(encoding="utf-8").splitlines()
        if stats_path.exists()
        else []
    )
    events = oom_counts_from_events(
        events_path.read_text(encoding="utf-8").splitlines()
        if events_path.exists()
        else []
    )
    rows = []
    for info in json.loads(_docker("inspect", *ids)):
        name = info["Name"].lstrip("/")
        running = info["State"]["Running"]
        peak = (
            _read_in_container(name, "/sys/fs/cgroup/memory.peak") if running else None
        )
        swap_peak = (
            _read_in_container(name, "/sys/fs/cgroup/memory.swap.peak")
            if running
            else None
        )
        mem_events = (
            _read_in_container(name, "/sys/fs/cgroup/memory.events")
            if running
            else None
        )
        kills = max(
            oom_kills_from_memory_events(mem_events or ""),
            events.get(name, 0),
            int(bool(info["State"].get("OOMKilled"))),
        )
        rows.append(
            ContainerMemory(
                name=name,
                service=info["Config"]["Labels"].get(
                    "com.docker.compose.service", name
                ),
                limit=info["HostConfig"].get("Memory") or 0,
                swap_limit=swap_allowance(info["HostConfig"]),
                sampled_peak=sampled.get(name),
                cgroup_peak=_int_file(peak),
                swap_peak=_int_file(swap_peak),
                oom_kills=kills,
                restarts=info.get("RestartCount", 0),
            )
        )
    return rows


# ── Commands ──────────────────────────────────────────────────────


def watch(out_dir: Path) -> int:
    """Record OOM events and sample docker stats until stopped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stopping = False

    def _stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    with (out_dir / EVENTS_FILE).open("a", encoding="utf-8") as events_out:
        events = subprocess.Popen(
            ["docker", "events", "--filter", "event=oom", "--format", "{{json .}}"],
            stdout=events_out,
        )
        try:
            with (out_dir / STATS_FILE).open("a", encoding="utf-8") as stats_out:
                while not stopping:
                    result = subprocess.run(
                        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        stats_out.write(result.stdout)
                        stats_out.flush()
                    time.sleep(SAMPLE_SECONDS)
        finally:
            events.terminate()
            events.wait(timeout=10)
    return 0


def report(out_dir: Path, project: str) -> int:
    """Print the table (and add it to the job summary); 1 if anything was OOM-killed."""
    rows = collect(project, out_dir)
    table = markdown_table(rows)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(table)
    killed = oom_killed(rows)
    if killed:
        print(f"OOM-killed: {', '.join(killed)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("watch", help="record OOM events and docker stats")
    w.add_argument("dir", type=Path)
    r = sub.add_parser("report", help="print the memory table, fail on OOM kills")
    r.add_argument("dir", type=Path)
    r.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args(argv)
    if args.command == "watch":
        return watch(args.dir)
    return report(args.dir, args.project)


if __name__ == "__main__":
    sys.exit(main())
