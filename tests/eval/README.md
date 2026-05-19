# RAG evaluation harness

Lightweight, runnable-on-a-laptop quality gate for the DocuSage RAG
pipeline. Without a number, "the redesign improved quality" is a feeling.

## What it measures

| Metric                | What it tells us                                                                   |
|-----------------------|------------------------------------------------------------------------------------|
| `retrieval_recall@k`  | Did the gold chunk appear in the top-k retrieved sources?                          |
| `citation_accuracy`   | Did the cited page match the expected page?                                        |
| `keyword_coverage`    | Fraction of expected keywords that appeared in the answer.                         |
| `faithfulness`        | LLM-as-judge: is the answer supported by the cited context? (yes / partial / no)   |
| `latency_p50` / `p95` | End-to-end SSE latency, in seconds.                                                |

## Layout

```
tests/eval/
  run_eval.py            # entry point — `python -m tests.eval.run_eval`
  fixtures/
    README.md            # drop your PDFs / DOCXs / etc. here
    qa.example.jsonl     # template question file (copy → qa.jsonl, edit)
```

## Setup

1. Drop your evaluation documents into `tests/eval/fixtures/`. Keep the
   set small (one or two PDFs) — this is a development-time gate, not
   a production benchmark.
2. Copy `qa.example.jsonl` to `qa.jsonl` and edit it. Each line is one
   `{question, expected_keywords, expected_doc, expected_page}` record.
3. Make sure the local stack is up (`make dev-up`) and the API is
   serving on `http://localhost:8000`. Set `EVAL_BASE_URL` to override.
4. Set `EVAL_USER_EMAIL` / `EVAL_USER_PASSWORD` to credentials for an
   account that exists on your local stack. The harness logs in via
   `/api/v1/auth/login` and reuses the access token.

## Run

```bash
make eval
# or:
python -m tests.eval.run_eval
```

The harness uploads every fixture, waits for `status=completed`, runs
every question against the streaming endpoint, scores each one, and
prints a summary table plus a JSON report at
`tests/eval/last_run.json`.

## Targets we want to clear after the redesign

For the I-9 sample dataset (~20 questions over `i-9instr.pdf`):

* `retrieval_recall@8` ≥ 0.85
* `citation_accuracy` ≥ 0.80
* `faithfulness` ≥ 0.85
* `latency_p95` ≤ 8 s on CPU with `qwen2.5:7b-instruct`

These are the bars the Phase 3 / Phase 4 work targets. A run below
these numbers is signal that something regressed — investigate before
shipping.
