"""
scripts/security/pip_audit_gate.py
──────────────────────────────────
Pass or fail on a pip-audit JSON report.

Fails when a pinned package has a known vulnerability with a released fix,
and names those packages so they can be upgraded. Vulnerabilities with no
fix yet are only counted: the log of a public repository is public, and
Dependabot alerts list them privately.

    pip-audit -r constraints.txt --no-deps --disable-pip -f json -o audit.json
    python scripts/security/pip_audit_gate.py audit.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    vuln_id: str
    fix_versions: tuple[str, ...]


def load_findings(report: dict) -> list[Finding]:
    """Flatten a pip-audit JSON report into one Finding per vulnerability."""
    return [
        Finding(
            package=dep["name"],
            version=dep.get("version", "?"),
            vuln_id=vuln["id"],
            fix_versions=tuple(vuln.get("fix_versions") or ()),
        )
        for dep in report.get("dependencies", [])
        for vuln in dep.get("vulns", [])
    ]


def summarize(findings: list[Finding]) -> tuple[list[str], int]:
    """Lines describing the fixable findings, and the count without a fix."""
    fixable = [f for f in findings if f.fix_versions]
    lines = [
        f"{f.package}=={f.version}: {f.vuln_id}, fixed in {', '.join(f.fix_versions)}"
        for f in sorted(fixable, key=lambda f: (f.package, f.vuln_id))
    ]
    return lines, len(findings) - len(fixable)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <pip-audit.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    fixable, unfixed = summarize(load_findings(report))
    for line in fixable:
        print(f"FAIL {line}")
    print(
        f"pip-audit: {len(fixable)} with a fix (must upgrade), "
        f"{unfixed} without a fix yet (see Dependabot alerts)"
    )
    return 1 if fixable else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
