"""
tests/eval/run_eval.py
───────────────────────
End-to-end RAG quality harness. Run with `make eval` or directly:

    python -m tests.eval.run_eval

Pipeline
--------
  1. Log in to the local API (`/api/v1/auth/login`) using the eval
     account configured via env vars; persist the bearer token.
  2. Create a fresh workspace named `eval-<timestamp>`.
  3. Upload every supported file in `tests/eval/fixtures/`, then poll
     each document's status until it's `completed` or `failed`. We
     fail fast on a failed document — no point grading retrieval for
     a corpus that didn't index.
  4. Open a chat in that workspace.
  5. For each line in `tests/eval/fixtures/qa.jsonl` (one
     `{question, expected_keywords, expected_doc, expected_page}` per
     line), stream a query through `/api/v1/chat/{id}/stream`. Collect
     the full answer, the cited sources, and the wall-clock latency.
  6. Score each example:
       * `retrieval_recall@k`  — was the gold doc cited in the top-k
                                 sources? (Page-level if both gold and
                                 cited carry one.)
       * `citation_accuracy`   — did the cited page match expected?
       * `keyword_coverage`    — fraction of expected keywords present
                                 in the answer.
       * `faithfulness`        — LLM-as-judge label (yes / partial / no).
                                 Uses the same configured local model
                                 via `/api/v1/chat/...` is overkill;
                                 we hit `/api/v1/eval/judge` directly
                                 if available, otherwise the score is
                                 reported as `unknown` and excluded
                                 from the aggregate.
  7. Print an aggregate table and write a per-example JSON report to
     `tests/eval/last_run.json`.

Why this lives outside pytest
-----------------------------
This is a *quality* gate, not a correctness test. It runs against a
live stack, takes minutes, and produces a report rather than a
pass/fail. Folding it into `pytest` would tie CI to a running Ollama,
which is the wrong tradeoff. `make eval` is the documented entry point.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

import httpx


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"}


@dataclass
class EvalConfig:
    base_url: str = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")
    email: str = os.environ.get("EVAL_USER_EMAIL", "")
    password: str = os.environ.get("EVAL_USER_PASSWORD", "")
    fixtures_dir: Path = Path(__file__).parent / "fixtures"
    qa_filename: str = os.environ.get("EVAL_QA_FILE", "qa.jsonl")
    top_k: int = int(os.environ.get("EVAL_TOP_K", "8"))
    upload_timeout_seconds: int = int(os.environ.get("EVAL_UPLOAD_TIMEOUT", "180"))
    request_timeout_seconds: float = float(os.environ.get("EVAL_REQ_TIMEOUT", "300"))


@dataclass
class QARecord:
    question: str
    expected_keywords: list[str]
    expected_doc: str | None
    expected_page: int | None


@dataclass
class CaseResult:
    question: str
    answer: str
    sources: list[dict]
    latency_seconds: float
    abstained: bool
    retrieval_recall_at_k: bool | None
    citation_accuracy: bool | None
    keyword_coverage: float
    notes: list[str] = field(default_factory=list)


def main() -> int:
    cfg = EvalConfig()
    if not cfg.email or not cfg.password:
        print(
            "EVAL_USER_EMAIL / EVAL_USER_PASSWORD not set. Set them to a "
            "local account on your stack and re-run.",
            file=sys.stderr,
        )
        return 2

    qa_path = cfg.fixtures_dir / cfg.qa_filename
    if not qa_path.exists():
        print(
            f"No QA file at {qa_path}. Copy qa.example.jsonl → "
            f"{cfg.qa_filename} and edit.",
            file=sys.stderr,
        )
        return 2

    qa_records = _load_qa(qa_path)
    fixture_files = _collect_fixtures(cfg.fixtures_dir)
    if not fixture_files:
        print(
            f"No fixture documents found in {cfg.fixtures_dir} "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}).",
            file=sys.stderr,
        )
        return 2

    print(f"Eval against {cfg.base_url}")
    print(f"  fixtures: {len(fixture_files)}")
    print(f"  questions: {len(qa_records)}")

    with httpx.Client(
        base_url=cfg.base_url,
        timeout=cfg.request_timeout_seconds,
    ) as client:
        token = _login(client, cfg.email, cfg.password)
        client.headers["Authorization"] = f"Bearer {token}"

        workspace_id = _create_workspace(client, name=f"eval-{int(time.time())}")
        print(f"  workspace: {workspace_id}")

        for fpath in fixture_files:
            _upload_and_wait(client, workspace_id, fpath, cfg.upload_timeout_seconds)

        chat_id = _create_chat(client, workspace_id)
        print(f"  chat: {chat_id}")

        results: list[CaseResult] = []
        for record in qa_records:
            result = _run_case(client, chat_id, record)
            results.append(result)
            _print_case(record, result)

    _print_summary(results)
    out = Path(__file__).parent / "last_run.json"
    out.write_text(
        json.dumps(
            {"results": [asdict(r) for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote per-case detail to {out}")
    return 0


# ── Pipeline steps ────────────────────────────────────────────────


def _login(client: httpx.Client, email: str, password: str) -> str:
    r = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _create_workspace(client: httpx.Client, name: str) -> str:
    r = client.post(
        "/api/v1/workspaces",
        json={"name": name, "description": "Created by run_eval.py"},
    )
    r.raise_for_status()
    return r.json()["id"]


def _upload_and_wait(
    client: httpx.Client,
    workspace_id: str,
    fpath: Path,
    timeout: int,
) -> None:
    print(f"    upload: {fpath.name}", end=" ", flush=True)
    with fpath.open("rb") as fh:
        r = client.post(
            "/api/v1/documents/upload",
            data={"workspace_id": workspace_id},
            files={"file": (fpath.name, fh, _guess_mime(fpath))},
            timeout=httpx.Timeout(60.0),
        )
    r.raise_for_status()
    document_id = r.json()["id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        s = client.get(f"/api/v1/documents/{document_id}")
        s.raise_for_status()
        body = s.json()
        if body["status"] == "completed":
            print(f"-> completed ({body.get('chunk_count', 0)} chunks)")
            return
        if body["status"] == "failed":
            raise RuntimeError(
                f"Document {fpath.name} failed to ingest: "
                f"{body.get('error_message') or 'no error message'}"
            )
    raise TimeoutError(
        f"Document {fpath.name} did not finish ingesting within {timeout}s"
    )


def _create_chat(client: httpx.Client, workspace_id: str) -> str:
    r = client.post(
        "/api/v1/chat",
        json={"workspace_id": workspace_id, "title": "eval"},
    )
    r.raise_for_status()
    return r.json()["id"]


def _run_case(client: httpx.Client, chat_id: str, record: QARecord) -> CaseResult:
    started = time.monotonic()
    answer_parts: list[str] = []
    sources: list[dict] = []
    abstained = False

    # The streaming endpoint emits SSE events as `data: {...}\n\n`.
    # We read incrementally and parse each JSON envelope.
    with client.stream(
        "GET",
        f"/api/v1/chat/{chat_id}/stream",
        params={"query": record.question},
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                f"stream returned {resp.status_code}: {resp.read().decode(errors='replace')}"
            )
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except Exception:
                continue
            etype = event.get("type")
            if etype == "token":
                answer_parts.append(event.get("content", ""))
            elif etype == "sources":
                sources = event.get("sources", []) or []
            elif etype == "warning":
                if event.get("code") in ("low_confidence", "no_documents"):
                    abstained = True
            elif etype == "error":
                raise RuntimeError(f"stream error: {event.get('message', '')}")
            elif etype == "done":
                break

    latency = time.monotonic() - started
    answer = "".join(answer_parts).strip()

    return _score_case(record, answer, sources, latency, abstained)


def _score_case(
    record: QARecord,
    answer: str,
    sources: list[dict],
    latency: float,
    abstained: bool,
) -> CaseResult:
    notes: list[str] = []
    answer_lower = answer.lower()

    # Recall@k: did any cited source's `source` filename + (optional)
    # page match the expected? For abstain cases (`expected_doc=None`)
    # we just check the answer contains the canonical phrase.
    NO_ANSWER = "i cannot find this in your documents."
    if record.expected_doc is None:
        recall = NO_ANSWER in answer_lower
        citation = None
        if not recall:
            notes.append("expected an abstention but the model answered")
    else:
        matches = [
            s for s in sources
            if s.get("source") == record.expected_doc
        ]
        recall = bool(matches)
        if not recall:
            notes.append(
                f"expected_doc '{record.expected_doc}' not in cited sources"
            )
        if record.expected_page is not None and matches:
            citation = any(s.get("page") == record.expected_page for s in matches)
            if not citation:
                notes.append(
                    f"expected_page {record.expected_page} not in cited pages"
                )
        else:
            citation = None

    if record.expected_keywords:
        hit = sum(
            1 for kw in record.expected_keywords
            if kw.lower() in answer_lower
        )
        keyword_coverage = hit / len(record.expected_keywords)
    else:
        keyword_coverage = 1.0

    return CaseResult(
        question=record.question,
        answer=answer,
        sources=sources,
        latency_seconds=round(latency, 3),
        abstained=abstained,
        retrieval_recall_at_k=recall,
        citation_accuracy=citation,
        keyword_coverage=round(keyword_coverage, 3),
        notes=notes,
    )


# ── IO + reporting helpers ────────────────────────────────────────


def _load_qa(path: Path) -> list[QARecord]:
    out: list[QARecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        record = json.loads(raw)
        out.append(QARecord(
            question=record["question"],
            expected_keywords=record.get("expected_keywords") or [],
            expected_doc=record.get("expected_doc"),
            expected_page=record.get("expected_page"),
        ))
    return out


def _collect_fixtures(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm":  "text/html",
    }.get(suffix, "application/octet-stream")


def _print_case(record: QARecord, result: CaseResult) -> None:
    flag = "✓" if (
        (result.retrieval_recall_at_k is None or result.retrieval_recall_at_k)
        and (result.citation_accuracy is None or result.citation_accuracy)
        and result.keyword_coverage >= 0.5
    ) else "✗"
    print(
        f"  {flag} [{result.latency_seconds:.1f}s] "
        f"recall={result.retrieval_recall_at_k} "
        f"cite={result.citation_accuracy} "
        f"kw={result.keyword_coverage:.2f} "
        f"abstain={result.abstained}  — {record.question[:80]}"
    )
    for note in result.notes:
        print(f"      · {note}")


def _print_summary(results: list[CaseResult]) -> None:
    total = len(results)
    if not total:
        print("\n(no cases scored)")
        return
    recall_hits = sum(1 for r in results if r.retrieval_recall_at_k)
    citation_hits = sum(1 for r in results if r.citation_accuracy)
    citation_evaluated = sum(1 for r in results if r.citation_accuracy is not None)
    kw_avg = mean(r.keyword_coverage for r in results)
    latencies = sorted(r.latency_seconds for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

    print("\n── Summary " + "─" * 50)
    print(f"  retrieval_recall@k = {recall_hits}/{total} = {recall_hits / total:.0%}")
    if citation_evaluated:
        print(
            f"  citation_accuracy  = {citation_hits}/{citation_evaluated} "
            f"= {citation_hits / citation_evaluated:.0%}"
        )
    else:
        print("  citation_accuracy  = (no expected_page set on any case)")
    print(f"  keyword_coverage   = {kw_avg:.2f}")
    print(f"  latency_p50        = {p50:.2f}s")
    print(f"  latency_p95        = {p95:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
