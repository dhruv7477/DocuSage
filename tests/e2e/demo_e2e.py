"""
End-to-end check of the demo stack (docker/docker-compose.demo.yml) through
Caddy, the way a visitor uses it:

  sign up -> upload a 40-page PDF and watch it ingest -> 5 uploads in a row
  all complete -> a streamed answer with sources.

CI also checks that only Caddy publishes ports (see .github/workflows/ci.yml).

Usage: python tests/e2e/demo_e2e.py [https://localhost]
Needs only httpx. Exits non-zero on the first failed check.
"""

import json
import sys
import time
import uuid

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://localhost"
API = f"{BASE}/api/v1"
CANARY_PAGE = 17
QUESTION = "What is the internal codename for the Q3 relocation project?"


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a small valid PDF with one text block per page (no dependencies)."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        text = " ".join(
            f"({line.replace('(', '[').replace(')', ']')}) Tj T*" for line in lines
        )
        stream = f"BT /F1 11 Tf 14 TL 72 740 Td {text} ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def handbook_pdf() -> bytes:
    pages = []
    for n in range(1, 41):
        lines = [
            f"Employee handbook, section {n}.",
            f"Policy {n}: requests are reviewed by the operations team within {n % 5 + 1} days.",
            "Staff should keep receipts and submit claims through the finance portal.",
        ]
        if n == CANARY_PAGE:
            lines.append(
                "The internal codename for the Q3 relocation project is HELIOTROPE-7741."
            )
        pages.append(lines)
    return make_pdf(pages)


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", flush=True)
        sys.exit(1)
    print(f"ok: {message}", flush=True)


def wait_for_documents(
    c: httpx.Client, h: dict, ids: dict[str, str], limit_s: int = 900
) -> dict[str, list[str]]:
    """Poll until every document is completed or failed; return statuses seen."""
    seen: dict[str, list[str]] = {name: [] for name in ids}
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        pending = False
        for name, doc_id in ids.items():
            status = c.get(f"{API}/documents/{doc_id}", headers=h).json()["status"]
            if not seen[name] or seen[name][-1] != status:
                seen[name].append(status)
            pending |= status not in ("completed", "failed")
        if not pending:
            return seen
        time.sleep(1)
    check(False, f"documents finished within {limit_s}s: {seen}")
    return seen


def stream_answer(c: httpx.Client, h: dict, chat_id: str) -> dict:
    events: list[dict] = []
    with c.stream(
        "GET",
        f"{API}/chat/{chat_id}/stream",
        params={"query": QUESTION},
        headers=h,
        timeout=300,
    ) as r:
        check(r.status_code == 200, f"stream returns 200 (got {r.status_code})")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events[-1].get("type") in ("done", "error"):
                    break
    return {
        "types": [e.get("type") for e in events],
        "answer": "".join(
            e.get("content", "") for e in events if e.get("type") == "token"
        ),
        "sources": next(
            (e["sources"] for e in events if e.get("type") == "sources"), []
        ),
    }


def main() -> None:
    with httpx.Client(timeout=120, verify=False, follow_redirects=False) as c:
        check(c.get(f"{BASE}/health").status_code == 200, "GET /health through Caddy")
        home = c.get(f"{BASE}/")
        check('<div id="root">' in home.text, "the frontend is served at /")
        deep_link = c.get(f"{BASE}/chat/some-id")
        check('<div id="root">' in deep_link.text, "deep links fall back to index.html")
        deep = c.get(f"{BASE}/health/deep")
        check("probes" not in deep.text, "/health/deep is not exposed through Caddy")
        plain = c.get(BASE.replace("https://", "http://") + "/")
        check(plain.status_code in (301, 302, 307, 308), "HTTP redirects to HTTPS")

        email = f"e2e.{uuid.uuid4().hex[:8]}@example.com"
        password = "E2ePass123!"
        r = c.post(
            f"{API}/auth/register",
            json={"email": email, "password": password, "full_name": "E2E"},
        )
        check(r.status_code == 201, f"sign up (got {r.status_code})")
        token = c.post(
            f"{API}/auth/login", json={"email": email, "password": password}
        ).json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}
        ws = c.post(f"{API}/workspaces", json={"name": "e2e"}, headers=h).json()["id"]

        pdf = handbook_pdf()
        notes = b"Remote work is allowed up to three days a week.\n" * 20
        uploads = [
            ("handbook.pdf", pdf),
            ("notes-1.txt", notes),
            ("handbook-copy.pdf", pdf),
            ("notes-2.txt", notes),
            ("handbook-copy-2.pdf", pdf),
        ]
        ids = {}
        for name, body in uploads:
            r = c.post(
                f"{API}/documents/upload",
                data={"workspace_id": ws},
                files={"file": (name, body)},
                headers=h,
            )
            check(r.status_code == 202, f"upload {name} accepted (got {r.status_code})")
            ids[name] = r.json()["id"]

        r = c.post(
            f"{API}/documents/upload",
            data={"workspace_id": ws},
            files={"file": ("one-too-many.txt", notes)},
            headers=h,
        )
        check(
            r.status_code == 429 and r.json().get("error_code") == "QUOTA_EXCEEDED",
            f"a sixth document is refused by the demo quota (got {r.status_code})",
        )

        seen = wait_for_documents(c, h, ids)
        for name, statuses in seen.items():
            check(
                statuses[-1] == "completed", f"{name} ingested: {' -> '.join(statuses)}"
            )
        check(
            any(len(s) > 2 for s in seen.values()),
            "an in-progress stage was visible while ingesting",
        )

        chat = c.post(f"{API}/chat", json={"workspace_id": ws}, headers=h).json()["id"]
        result = stream_answer(c, h, chat)
        check(
            result["types"][-1] == "done", f"stream ends with done: {result['types']}"
        )
        check(
            result["types"].count("token") > 1, "answer arrives as several token events"
        )
        check("[1]" in result["answer"], f"answer cites a source: {result['answer']!r}")
        pages = {s.get("page") for s in result["sources"]}
        check(
            CANARY_PAGE in pages,
            f"sources include the canary page {CANARY_PAGE}: {pages}",
        )

    print("demo end-to-end check passed", flush=True)


if __name__ == "__main__":
    main()
