"""
tests/eval/decline/setup_account.py
-----------------------------------
Creates (or reuses) the decline eval account on a running demo stack,
through the public HTTP API only, so ingestion runs through the real
worker exactly as it does for a visitor:

1. register the account (an existing account is reused),
2. create a workspace for each synthetic resume ("Decline eval: resume"
   and "Decline eval: resume 2") and upload the resume into it,
3. wait until the resumes and the account's "Sample documents" copies of
   the I-9 and NIST samples are all completed.

Usage (host Python with httpx). Caddy signs https://localhost with its own
local CA; point E2E_CA_CERT at its root certificate (see README.md):

    DECLINE_EVAL_PASSWORD=... python tests/eval/decline/setup_account.py \
        [--base-url https://localhost] [--email decline-eval@example.com]

The password must have an uppercase letter and a digit (the API's rule).
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
# Each synthetic resume in a workspace of its own.
RESUMES = {
    "Decline eval: resume": HERE / "resume_talia_brennerholt.pdf",
    "Decline eval: resume 2": HERE / "resume_rosalind_kettleborne.pdf",
}
SAMPLES_WORKSPACE = "Sample documents"
EXPECTED_SAMPLE_COUNT = 2
POLL_SECONDS = 5.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="https://localhost")
    parser.add_argument("--email", default="decline-eval@example.com")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds")
    return parser.parse_args()


def _login(client: httpx.Client, email: str, password: str) -> str:
    """Register the account if it does not exist, then return a bearer token."""
    register = client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": "Decline Eval"},
    )
    if register.status_code not in (201, 400, 409):
        register.raise_for_status()
    login = client.post("/auth/login", json={"email": email, "password": password})
    login.raise_for_status()
    return login.json()["access_token"]


def _workspaces(client: httpx.Client) -> dict[str, str]:
    """Map workspace name to id for the signed-in account."""
    response = client.get("/workspaces")
    response.raise_for_status()
    return {ws["name"]: ws["id"] for ws in response.json()}


def _documents(client: httpx.Client, workspace_id: str) -> list[dict]:
    response = client.get("/documents", params={"workspace_id": workspace_id})
    response.raise_for_status()
    return response.json()["documents"]


def _ensure_resume(client: httpx.Client, workspace: str, pdf: Path) -> str:
    """Create a resume workspace and upload its PDF unless already there."""
    workspace_id = _workspaces(client).get(workspace)
    if workspace_id is None:
        created = client.post("/workspaces", json={"name": workspace})
        created.raise_for_status()
        workspace_id = created.json()["id"]
    names = {d["original_filename"] for d in _documents(client, workspace_id)}
    if pdf.name not in names:
        with pdf.open("rb") as fh:
            upload = client.post(
                "/documents/upload",
                data={"workspace_id": workspace_id},
                files={"file": (pdf.name, fh, "application/pdf")},
            )
        upload.raise_for_status()
    return workspace_id


def _wait_until_ready(client: httpx.Client, deadline: float) -> dict[str, str]:
    """Poll until every eval workspace holds only completed documents."""
    expected = {name: 1 for name in RESUMES} | {
        SAMPLES_WORKSPACE: EXPECTED_SAMPLE_COUNT
    }
    while True:
        spaces = _workspaces(client)
        ready: dict[str, str] = {}
        for name, minimum in expected.items():
            workspace_id = spaces.get(name)
            if workspace_id is None:
                continue
            docs = _documents(client, workspace_id)
            failed = [d["original_filename"] for d in docs if d["status"] == "failed"]
            if failed:
                raise RuntimeError(f"Ingestion failed in {name!r}: {failed}")
            done = [d for d in docs if d["status"] == "completed"]
            if len(done) >= minimum and len(done) == len(docs):
                ready[name] = workspace_id
        if len(ready) == len(expected):
            return ready
        if time.monotonic() > deadline:
            raise TimeoutError(f"Workspaces not ready in time; ready so far: {ready}")
        time.sleep(POLL_SECONDS)


def main() -> int:
    """Set up the account and print the eval workspace ids."""
    args = _parse_args()
    password = os.environ.get("DECLINE_EVAL_PASSWORD")
    if not password:
        print("Set DECLINE_EVAL_PASSWORD first.", file=sys.stderr)
        return 2
    base = args.base_url.rstrip("/") + "/api/v1"
    # Verify the server certificate, against Caddy's local CA when
    # E2E_CA_CERT points to it (as in tests/e2e).
    tls = ssl.create_default_context(cafile=os.environ.get("E2E_CA_CERT"))
    with httpx.Client(base_url=base, verify=tls, timeout=60.0) as client:
        token = _login(client, args.email, password)
        client.headers["Authorization"] = f"Bearer {token}"
        for workspace, pdf in RESUMES.items():
            _ensure_resume(client, workspace, pdf)
        ready = _wait_until_ready(client, time.monotonic() + args.timeout)
    for name, workspace_id in ready.items():
        print(f"{name}: {workspace_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
