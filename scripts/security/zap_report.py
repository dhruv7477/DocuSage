"""
scripts/security/zap_report.py
──────────────────────────────
Pass or fail on OWASP ZAP JSON reports, and convert them to SARIF for
GitHub's Security tab.

Fails when any alert is High risk. Medium, Low and Informational alerts are
counted and listed by rule, and go to the Security tab through the SARIF
file. Rules ZAP was told to ignore (tests/security/zap/rules.tsv) are not in
the reports at all.

    python scripts/security/zap_report.py --sarif zap.sarif baseline.json api.json

SARIF needs a file in the repository for every result. ZAP reports URLs, so
an API URL points at src/main.py and anything else at the Caddyfile, which
serves the frontend and sets the response headers. The URL, method and
parameter are in the message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

RISK_NAMES = {3: "High", 2: "Medium", 1: "Low", 0: "Informational"}
SARIF_LEVELS = {3: "error", 2: "warning", 1: "note", 0: "note"}
APP_FILE = "src/main.py"
EDGE_FILE = "docker/caddy/Caddyfile"


@dataclass(frozen=True)
class Alert:
    rule_id: str
    name: str
    risk: int
    method: str
    uri: str
    param: str
    evidence: str
    solution: str
    reference: str


def load_alerts(report: dict) -> list[Alert]:
    """One Alert per instance in a ZAP traditional-json report."""
    alerts: list[Alert] = []
    for site in report.get("site", []):
        for alert in site.get("alerts", []):
            for instance in alert.get("instances") or [{}]:
                alerts.append(
                    Alert(
                        rule_id=str(alert.get("pluginid", "")),
                        name=alert.get("name") or alert.get("alert", ""),
                        risk=int(alert.get("riskcode", 0)),
                        method=instance.get("method", ""),
                        uri=instance.get("uri", ""),
                        param=instance.get("param", ""),
                        evidence=instance.get("evidence", ""),
                        solution=_strip_html(alert.get("solution", "")),
                        reference=_strip_html(alert.get("reference", "")),
                    )
                )
    return alerts


def summary_lines(alerts: list[Alert]) -> list[str]:
    """Counts per risk, then each rule with its risk and instance count."""
    by_risk = Counter(a.risk for a in alerts)
    lines = [
        "ZAP: "
        + ", ".join(f"{by_risk.get(r, 0)} {RISK_NAMES[r]}" for r in (3, 2, 1, 0))
    ]
    by_rule = Counter((a.risk, a.rule_id, a.name) for a in alerts)
    for (risk, rule_id, name), count in sorted(by_rule.items(), reverse=True):
        lines.append(f"  {RISK_NAMES[risk]:<13} {rule_id:>6}  {name} ({count})")
    return lines


def to_sarif(alerts: list[Alert]) -> dict:
    """A SARIF 2.1.0 log with one rule per ZAP rule and one result per instance."""
    rules: dict[str, dict] = {}
    results = []
    for alert in alerts:
        rules.setdefault(
            alert.rule_id,
            {
                "id": alert.rule_id,
                "name": alert.name,
                "shortDescription": {"text": alert.name},
                "help": {"text": alert.solution or alert.name},
                "helpUri": f"https://www.zaproxy.org/docs/alerts/{alert.rule_id}/",
                "properties": {"security-severity": _security_severity(alert.risk)},
            },
        )
        results.append(
            {
                "ruleId": alert.rule_id,
                "level": SARIF_LEVELS[alert.risk],
                "message": {"text": _message(alert)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": _file_for(alert.uri)},
                            "region": {"startLine": 1},
                        }
                    }
                ],
                "partialFingerprints": {"zapInstance/v1": _fingerprint(alert)},
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "OWASP ZAP",
                        "informationUri": "https://www.zaproxy.org/",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }


def _file_for(uri: str) -> str:
    return APP_FILE if urlsplit(uri).path.startswith("/api/") else EDGE_FILE


def _message(alert: Alert) -> str:
    parts = [f"{alert.name}: {alert.method} {alert.uri}".strip()]
    if alert.param:
        parts.append(f"parameter {alert.param}")
    if alert.evidence:
        parts.append(f"evidence {alert.evidence[:200]}")
    return ", ".join(parts)


def _fingerprint(alert: Alert) -> str:
    key = "|".join((alert.rule_id, alert.method, alert.uri, alert.param))
    return hashlib.sha256(key.encode()).hexdigest()


def _security_severity(risk: int) -> str:
    # GitHub maps 7.0 and up to High, 4.0 to Medium, below that to Low.
    return {3: "7.5", 2: "5.0", 1: "3.0", 0: "0.0"}[risk]


def _strip_html(text: str) -> str:
    return text.replace("<p>", "").replace("</p>", "\n").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate and convert ZAP JSON reports.")
    parser.add_argument("reports", nargs="+", type=Path, help="ZAP JSON reports")
    parser.add_argument("--sarif", type=Path, help="write a SARIF file here")
    args = parser.parse_args(argv)

    alerts = [
        alert
        for path in args.reports
        for alert in load_alerts(json.loads(path.read_text(encoding="utf-8")))
    ]
    for line in summary_lines(alerts):
        print(line)
    if args.sarif:
        args.sarif.write_text(json.dumps(to_sarif(alerts), indent=2), encoding="utf-8")

    high = sum(1 for a in alerts if a.risk == 3)
    if high:
        print(f"FAIL: {high} High risk alert instance(s)")
        return 1
    print("PASS: no High risk alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
