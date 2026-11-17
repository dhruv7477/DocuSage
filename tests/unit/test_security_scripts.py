"""
tests/unit/test_security_scripts.py
───────────────────────────────────
The CI security gates: pip-audit and ZAP report handling, and the OpenAPI
export for the ZAP API scan.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.security import export_openapi, pip_audit_gate, zap_report

pytestmark = pytest.mark.unit


# ── pip-audit gate ────────────────────────────────────────────────


def _audit(*deps: tuple[str, str, list[list[str]]]) -> dict:
    return {
        "dependencies": [
            {
                "name": name,
                "version": version,
                "vulns": [
                    {"id": f"PYSEC-{i}", "fix_versions": fixes}
                    for i, fixes in enumerate(vulns)
                ],
            }
            for name, version, vulns in deps
        ]
    }


def _run_gate(tmp_path: Path, report: dict) -> int:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return pip_audit_gate.main(["gate", str(path)])


def test_pip_audit_gate_passes_a_clean_report(tmp_path: Path, capsys) -> None:
    assert _run_gate(tmp_path, _audit(("fastapi", "0.162.0", []))) == 0
    assert "0 with a fix" in capsys.readouterr().out


def test_pip_audit_gate_fails_on_a_fixable_vulnerability(
    tmp_path: Path, capsys
) -> None:
    code = _run_gate(tmp_path, _audit(("pillow", "10.4.0", [["12.1.1"]])))
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL pillow==10.4.0: PYSEC-0, fixed in 12.1.1" in out


def test_pip_audit_gate_only_counts_vulnerabilities_without_a_fix(
    tmp_path: Path, capsys
) -> None:
    code = _run_gate(tmp_path, _audit(("somepkg", "1.0", [[], []])))
    out = capsys.readouterr().out
    assert code == 0
    assert "2 without a fix" in out
    assert "PYSEC" not in out  # unfixed IDs stay out of public logs


# ── ZAP report ────────────────────────────────────────────────────


def _zap(*alerts: tuple[str, int, list[str]]) -> dict:
    return {
        "site": [
            {
                "@name": "https://localhost",
                "alerts": [
                    {
                        "pluginid": rule_id,
                        "name": f"rule {rule_id}",
                        "riskcode": str(risk),
                        "solution": "<p>Fix it.</p>",
                        "instances": [{"uri": uri, "method": "GET"} for uri in uris],
                    }
                    for rule_id, risk, uris in alerts
                ],
            }
        ]
    }


def _run_zap(tmp_path: Path, report: dict) -> tuple[int, dict]:
    report_path = tmp_path / "zap.json"
    sarif_path = tmp_path / "zap.sarif"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    code = zap_report.main(["--sarif", str(sarif_path), str(report_path)])
    return code, json.loads(sarif_path.read_text(encoding="utf-8"))


def test_zap_report_fails_on_high_risk(tmp_path: Path, capsys) -> None:
    code, _ = _run_zap(tmp_path, _zap(("40018", 3, ["https://localhost/api/v1/x"])))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_zap_report_passes_medium_and_lists_it(tmp_path: Path, capsys) -> None:
    code, _ = _run_zap(tmp_path, _zap(("10055", 2, ["https://localhost/"])))
    out = capsys.readouterr().out
    assert code == 0
    assert "0 High, 1 Medium" in out
    assert "10055  rule 10055 (1)" in out


def test_zap_sarif_has_one_result_per_instance(tmp_path: Path) -> None:
    _, sarif = _run_zap(
        tmp_path,
        _zap(
            ("10055", 2, ["https://localhost/", "https://localhost/api/v1/demo"]),
            ("10096", 0, ["https://localhost/assets/a.js"]),
        ),
    )
    run = sarif["runs"][0]
    results = run["results"]
    assert sarif["version"] == "2.1.0"
    assert [r["id"] for r in run["tool"]["driver"]["rules"]] == ["10055", "10096"]
    assert [r["level"] for r in results] == ["warning", "warning", "note"]
    files = [
        r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for r in results
    ]
    assert files == ["docker/caddy/Caddyfile", "src/main.py", "docker/caddy/Caddyfile"]
    fingerprints = {r["partialFingerprints"]["zapInstance/v1"] for r in results}
    assert len(fingerprints) == 3


# ── OpenAPI export and production docs ───────────────────────────


def test_openapi_export_targets_the_scan_server_without_logout() -> None:
    schema = export_openapi.build_schema(
        "https://localhost", export_openapi.SESSION_ENDING_PATHS
    )
    assert schema["servers"] == [{"url": "https://localhost"}]
    assert "/api/v1/auth/logout" not in schema["paths"]
    assert "/api/v1/auth/login" in schema["paths"]


def test_production_serves_no_api_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import main

    monkeypatch.setattr(main.settings, "app_env", "production")
    client = TestClient(main.create_app())  # no lifespan: no service probes

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path
