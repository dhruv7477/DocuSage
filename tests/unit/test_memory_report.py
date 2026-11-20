"""
tests/unit/test_memory_report.py
────────────────────────────────
The parsing and the OOM decision in tests/e2e/memory_report.py, which the
demo end-to-end job runs. Docker itself is not called.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
import memory_report as mr  # noqa: E402

pytestmark = pytest.mark.unit

MIB = 1024**2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("512MiB", 512 * MIB),
        ("1.5GiB", round(1.5 * 1024**3)),
        ("812.3kB", 812_300),
        ("0B", 0),
        (" 64MiB ", 64 * MIB),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert mr.parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "12", "MiB", "3 parsecs"])
def test_parse_size_rejects_other_text(text: str) -> None:
    with pytest.raises(ValueError):
        mr.parse_size(text)


def _stats(name: str, used: str) -> str:
    return json.dumps({"Name": name, "MemUsage": f"{used} / 640MiB"})


def test_peaks_from_stats_keeps_the_highest_sample_per_container() -> None:
    lines = [
        _stats("api", "300MiB"),
        _stats("worker", "200MiB"),
        "",
        _stats("api", "450MiB"),
        _stats("api", "420MiB"),
    ]
    assert mr.peaks_from_stats(lines) == {"api": 450 * MIB, "worker": 200 * MIB}


def test_peaks_from_stats_skips_containers_without_a_reading() -> None:
    starting = json.dumps({"Name": "e2e", "MemUsage": "-- / --"})
    assert mr.peaks_from_stats([starting, _stats("api", "300MiB")]) == {
        "api": 300 * MIB
    }


def test_oom_counts_from_events() -> None:
    lines = [
        json.dumps({"Action": "oom", "Actor": {"Attributes": {"name": "worker"}}}),
        json.dumps({"Action": "oom", "Actor": {"Attributes": {"name": "worker"}}}),
        json.dumps({"Action": "die", "Actor": {"Attributes": {"name": "api"}}}),
        "",
    ]
    assert mr.oom_counts_from_events(lines) == {"worker": 2}


def test_oom_kills_from_memory_events() -> None:
    text = "low 0\nhigh 0\nmax 12\noom 3\noom_kill 2\noom_group_kill 0\n"
    assert mr.oom_kills_from_memory_events(text) == 2
    assert mr.oom_kills_from_memory_events("") == 0


def _row(
    service: str,
    limit_mib: int,
    peak_mib: int | None,
    kills: int = 0,
    swap_mib: int = 0,
):
    return mr.ContainerMemory(
        name=f"docusage-demo-{service}-1",
        service=service,
        limit=limit_mib * MIB,
        swap_limit=swap_mib * MIB,
        sampled_peak=None if peak_mib is None else peak_mib * MIB,
        cgroup_peak=None,
        swap_peak=None,
        oom_kills=kills,
        restarts=0,
    )


def test_markdown_table_lists_every_service_and_the_total() -> None:
    table = mr.markdown_table(
        [_row("worker", 512, 400), _row("api", 640, 500, swap_mib=512)]
    )
    lines = table.splitlines()
    api = next(i for i, line in enumerate(lines) if line.startswith("| api |"))
    worker = next(i for i, line in enumerate(lines) if line.startswith("| worker |"))
    assert api < worker  # sorted by service
    assert "| api | 640 MiB | 512 MiB | 500 MiB | n/a | n/a | 0 | 0 |" in lines
    assert "| **total** | **1152 MiB** | **512 MiB** | **900 MiB** | | | | |" in lines


def test_markdown_table_shows_a_missing_limit() -> None:
    assert "| mock-llm | none | 0 MiB | n/a |" in mr.markdown_table(
        [_row("mock-llm", 0, None)]
    )


@pytest.mark.parametrize(
    ("host_config", "expected"),
    [
        ({"Memory": 600 * MIB, "MemorySwap": 1112 * MIB}, 512 * MIB),
        ({"Memory": 64 * MIB, "MemorySwap": 64 * MIB}, 0),
        ({"Memory": 0, "MemorySwap": 0}, 0),
        ({"Memory": 64 * MIB, "MemorySwap": -1}, 0),
    ],
)
def test_swap_allowance(host_config: dict, expected: int) -> None:
    assert mr.swap_allowance(host_config) == expected


def test_oom_killed_lists_only_services_with_kills() -> None:
    rows = [_row("api", 640, 500), _row("worker", 512, 510, kills=1)]
    assert mr.oom_killed(rows) == ["worker"]
    assert mr.oom_killed([_row("api", 640, 500)]) == []


def test_report_fails_and_writes_the_summary_on_an_oom_kill(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    rows = [_row("api", 640, 500), _row("worker", 512, 510, kills=1)]
    monkeypatch.setattr(mr, "collect", lambda project, out_dir: rows)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert mr.report(tmp_path, "docusage-demo") == 1

    assert "| worker | 512 MiB |" in summary.read_text(encoding="utf-8")
    assert "OOM-killed: worker" in capsys.readouterr().err


def test_report_passes_without_oom_kills(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mr, "collect", lambda project, out_dir: [_row("api", 640, 500)])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert mr.report(tmp_path, "docusage-demo") == 0
