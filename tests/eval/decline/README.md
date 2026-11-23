# Decline eval

Measures one decision: does the chat pipeline answer a question, or does it
send the fixed reply "I couldn't find anything about that in your
documents..."? That choice is made before any LLM call (router, retrieval,
cross-encoder rerank, answer gate), so this eval makes **no LLM calls at
all**. It is safe to run against a stack that shares a small hosted LLM
quota. CI runs it on every pull request (the "Demo stack end to end" job)
and fails on any regression (below).

## Files

| File | What it is |
|---|---|
| `make_resume_pdf.py` | Builds `resume_talia_brennerholt.pdf` with PyMuPDF. |
| `resume_talia_brennerholt.pdf` | A two-page synthetic resume: a cover letter, then the resume. The person, companies, school and addresses are invented; emails use example.com. The name is in the largest font, so the PDF parser makes it a heading, as with a real resume. |
| `questions.jsonl` | 51 labelled questions in five categories (below). |
| `setup_account.py` | Creates the eval account through the HTTP API, uploads the resume, and waits for ingestion and the sample copies. |
| `run_decline_eval.py` | The harness. |

The other two documents are the demo samples in `demo/samples/` (the I-9
instructions and NIST CSF 2.0). Every demo account gets copies of them in
its "Sample documents" workspace.

## Question set

Each line has `id`, `category`, `workspace`, `doc`, `question`, `note` (the
expected answer, or why the question must be declined) and `split`.

| Category | Expected | Count | Examples |
|---|---|---|---|
| `answerable_specific` | answer | 18 | Entity lookups ("What is the name of the candidate?", "When is the candidate available to start?"), lists ("Which documents are mentioned in this?"), facts ("When must an employer complete Section 2 of Form I-9?") |
| `answerable_vague` | answer | 8 | "Who is this person?", "What is this framework for?" |
| `summary` | answer | 8 | "Summary of the document", "TL;DR", "What are the main takeaways from these documents?" |
| `near_topic` | decline | 9 | "What is Canada's equivalent of Form I-9?", "What salary is Corvantis Labs offering for this role?" |
| `off_topic` | decline | 8 | "Summarize the plot of Hamlet", "What do they call a baby kangaroo?" |

`workspace` is `resume` (the "Decline eval: resume" workspace, which holds
only the resume) or `samples` (the "Sample documents" workspace, I-9 and
NIST together). Near-topic questions sit close to a document's subject but
the documents do not answer them. The off-topic set includes requests that
use the word "summary" and two that use "they" or "this" without referring
to the documents.

`split` is `holdout` for a third of each category and `tune` for the rest.
It was drawn once (stratified by category, `random.Random(20261123)`) and
is stored in the file so it never changes when questions are added. Pick
thresholds on `tune` only and report `holdout`.

## How to run

The commands assume the local demo stack is up as compose project
`docusage-answers` and run from the repository root in Git Bash. Adjust
`-p` if your project name differs.

1. Create the account once (host Python with httpx). The password needs an
   uppercase letter and a digit. Caddy signs https://localhost with its own
   local CA, so copy its root certificate out first and trust it.

   ```bash
   docker compose -p docusage-answers -f docker/docker-compose.demo.yml \
     cp caddy:/data/caddy/pki/authorities/local/root.crt caddy-root.crt
   E2E_CA_CERT=caddy-root.crt DECLINE_EVAL_PASSWORD='<a password>' \
     python tests/eval/decline/setup_account.py
   ```

2. Run the harness in a one-off container from the api image. It mounts
   this folder at `/work` and uses the image's app code at `/src`.
   `OPENAI_BASE_URL` and `OPENAI_API_KEY` are overridden so that even a
   call that slipped past the guard could not reach the hosted LLM.

   ```bash
   MSYS_NO_PATHCONV=1 docker compose -p docusage-answers \
     -f docker/docker-compose.demo.yml run --rm --no-deps \
     -e PYTHONPATH=/src -e LOG_LEVEL=WARNING \
     -e OPENAI_BASE_URL=http://127.0.0.1:9/v1 -e OPENAI_API_KEY=decline-eval-guard \
     --entrypoint python -v "$(pwd)/tests/eval/decline:/work:ro" \
     api /work/run_decline_eval.py
   ```

   Add `--json /some/mounted/dir/result.json` to keep the full results.
   Any setting can be tried with `-e`, for example
   `-e ABSTENTION_RERANKER_THRESHOLD=... -e ABSTENTION_COSINE_THRESHOLD=...`.

   To test code that is not in the image yet, also mount the working tree's
   `src`: `-v "$(pwd)/src:/src/src:ro"`.

3. To rebuild the PDF after editing the generator:

   ```bash
   MSYS_NO_PATHCONV=1 docker compose -p docusage-answers \
     -f docker/docker-compose.demo.yml run --rm --no-deps \
     --entrypoint python -v "$(pwd)/tests/eval/decline:/work" \
     api /work/make_resume_pdf.py
   ```

   Re-upload it afterwards (delete the old copy in the resume workspace,
   then run `setup_account.py` again).

## What the harness does

For each question it calls the app's real orchestrator,
`_get_orchestrator(_get_rag_service())` from `src/api/v1/chat.py`, with
`prepare(...)`, exactly as the SSE endpoint does: single turn, no history,
no document filter, and the workspace's completed documents. It reads
`plan.intent`; `off_topic` means declined. Near-topic and off-topic
questions are expected to be declined; everything else is expected to be
answered. It never reads
`plan.token_iter`, so no answer is generated.

Before anything runs, the LLM factory is patched to build a guard provider
whose `generate` and `stream` record the attempt and raise. The report ends
with the number of attempted LLM calls, which must be 0.

The per-question table shows the route events, the confidence, the top
three reranker scores (sigmoid), the best dense cosine score and the
outcome. The scores are diagnostics only; the verdict is the plan's
intent, so the harness keeps working when the decision logic changes.

After the table, the harness prints three summaries (all questions, `tune`
and `holdout`): the decline rate per category, the false declines
(answerable or summary questions that got the fixed reply), the near-topic
or off-topic questions sent to the summary route (which never declines),
those sent to the LLM (whose grounded prompt lets it decline; this
harness cannot see that verdict), and how many questions reach the LLM.

The exit code is 1 on any regression: an LLM call, an error, a false
decline, or a near-topic or off-topic request sent to the summary route.
Those are the parts of the decision that need no LLM to check.

## Baseline on master (ad37336)

Demo settings: `cross-encoder/ms-marco-MiniLM-L-6-v2`, mean of the top 3
sigmoid scores, threshold 0.3, summaries off, router LLM off.

| Category | Declined (all 51) | Declined (holdout 18) |
|---|---|---|
| answerable_specific | 8/18 | 3/6 |
| answerable_vague | 7/8 | 2/3 |
| summary | 8/8 | 3/3 |
| near_topic | 8/9 | 3/3 |
| off_topic | 8/8 | 3/3 |

All 51: 23 false declines, 1 false answer ("What is Canada's equivalent of
Form I-9?"), 12 of 51 questions reach the LLM. LLM calls attempted: 0.

## The answer gate (this change)

The cross-encoder ranks well but scores answers worded unlike the question
near zero: against the same resume chunk, "How many years of experience
does the data engineer have?" scores 1.000 and "... does the candidate
have?" 0.027. So it no longer decides alone. A question is declined
without an LLM call only when the top reranker score is below
`ABSTENTION_RERANKER_THRESHOLD` (0.5) **and** the top dense cosine is below
`ABSTENTION_COSINE_THRESHOLD` (0.48). Everything else goes to the LLM.
Summary requests that name nothing but the documents get a one-call
overview; one with its own subject ("Summarize the plot of Hamlet") goes
through the gate.

How the cut-offs were chosen:

- Reranker: every value from 0.001 to 1.0 gives the same result on this
  set; 0.5 means a strong reranker hit always vetoes a cosine decline.
- Cosine, on `tune` only: the highest must-decline cosine below the
  answerable questions is 0.4957 and the lowest answerable cosine is
  0.5171; the midpoint is 0.5065. On `holdout` the closest answerable
  question ("What kind of work does this person do?") is at 0.507, only
  0.0005 above it.
- Then a real one-page resume (not in this set) was asked "What is the name
  of the candidate?": top cosine 0.4979, reranker 0.0, declined at 0.5065.
  Entity questions on short documents score near any cosine cut-off, so
  the default is 0.48. That moves four more must-decline questions
  (cosines 0.488 to 0.496) from the gate to the LLM.

At 0.48:

| Category | Declined by the gate (all 51) | Declined by the gate (holdout 18) |
|---|---|---|
| answerable_specific | 0/18 | 0/6 |
| answerable_vague | 0/8 | 0/3 |
| summary | 0/8 | 0/3 |
| near_topic | 0/9 | 0/3 |
| off_topic | 5/8 | 2/3 |

All 51: 0 false declines, 0 requests sent to the wrong route, 46 of 51
questions reach the LLM. The 12 near-topic and off-topic questions that
reach it were each sent once to gpt-oss-20b with the app's grounded
prompt: 11 were declined ("I cannot find this in your documents."). The
twelfth, "How do I write a good cover letter?" against the resume, got
general advice from gpt-oss-20b but was declined by gpt-oss-120b, the
model the demo answers with. The answerable questions sent to gpt-oss-20b
were all answered.
