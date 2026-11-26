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
| `make_resume_pdf.py` | Builds both synthetic resumes with PyMuPDF. |
| `resume_talia_brennerholt.pdf` | A two-page synthetic resume: a cover letter, then the resume. The person, companies, school and addresses are invented; emails use example.com. The name is in the largest font, so the PDF parser makes it a heading, as with a real resume. |
| `resume_rosalind_kettleborne.pdf` | A three-page synthetic resume whose name appears only as its top heading and on page 1 (17 chunks). Page 2 has a fact stated only in a heading: "First place, Riverbend Data Jam 2024" is a subheading under Hackathons. |
| `questions.jsonl` | 56 labelled questions in six categories (below). |
| `setup_account.py` | Creates the eval account through the HTTP API, uploads each resume into its own workspace, and waits for ingestion and the sample copies. |
| `run_decline_eval.py` | The harness. |

The other two documents are the demo samples in `demo/samples/` (the I-9
instructions and NIST CSF 2.0). Every demo account gets copies of them in
its "Sample documents" workspace.

## Question set

Each line has `id`, `category`, `workspace`, `doc`, `question`, `note` (the
expected answer, or why the question must be declined) and `split`.
Sixteen questions also have `answer`: the exact text a cited chunk must
contain (names, a role, dates, tools, a school, a phone number, a talk).

| Category | Expected | Count | Examples |
|---|---|---|---|
| `answerable_specific` | answer | 21 | Entity lookups ("What is the name of the candidate?", "When is the candidate available to start?"), lists ("Which documents are mentioned in this?"), facts ("When must an employer complete Section 2 of Form I-9?") |
| `answerable_vague` | answer | 9 | "Who is this person?", "What is this framework for?", "Tell me about the candidate." |
| `summary` | answer | 8 | "Summary of the document", "TL;DR", "What are the main takeaways from these documents?" |
| `near_topic` | decline | 9 | "What is Canada's equivalent of Form I-9?", "What salary is Corvantis Labs offering for this role?" |
| `off_topic` | decline | 8 | "Summarize the plot of Hamlet", "What do they call a baby kangaroo?" |
| `general_knowledge` | answer | 1 | "What does a machine learning engineer typically do?" against the second resume |

`workspace` is `resume` or `resume2` (the "Decline eval: resume" and
"Decline eval: resume 2" workspaces, each holding one resume) or `samples`
(the "Sample documents" workspace, I-9 and NIST together). Near-topic questions sit close to a document's subject but
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

   Re-upload them afterwards (delete the old copy in its resume workspace,
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

**Recall.** For each question with an `answer`, the harness reports where
the first chunk whose page text contains it ranks in dense search and in
the prompt's context. The answer must be in an included chunk.

**Citation accuracy.** A model that cites any included chunk for the
answer is simulated: `"<answer> [k]."` for every k, run through the app's
citation rule (`finalize_rag_answer` in
`src/services/citation_grounding.py`). Every citation it shows must point
at a chunk whose page text contains the answer; a heading path never
counts. This needs no LLM, and catches the bugs found on the live demo: the
prompt shows each chunk's heading path, so a model can cite any chunk
under the resume's name heading for the name.

**General knowledge.** For a `general_knowledge` question, an answer that
cites nothing must be shown with the "Not from your documents" label and
no sources.

**Cost and small workspaces.** For each question that reaches the answer
LLM, the harness reports the prompt's estimated tokens (the app's
tiktoken count) per workspace. In a small workspace (below), each
document's first chunk must be in the prompt.

The exit code is 1 on any regression: an LLM call, an error, a false
decline, a near-topic or off-topic request sent to the summary route, an
entity answer that no included chunk states, a shown citation that does
not land on a chunk stating the answer, an uncited answer without the
label, or a document's first chunk missing from a small workspace's
prompt. Those are the parts of the decision that need no LLM to check.

| Citation accuracy (entity lookups) | Citations landing on a chunk that states the answer |
|---|---|
| Master (citations stay where the model put them) | 5 of 32 |
| With citation grounding and headings in chunk text | 32 of 32 |

The same four questions asked of gpt-oss-120b through the stream endpoint
(4 calls): every citation in the final answers points at a chunk that
states the answer, and its page image highlights it.

## The citation rule and document titles

A citation is shown only when the cited chunk's page text supports the
claim. One that doesn't moves to an included chunk that does, or is
dropped. An answer whose citations are all dropped says "I cannot find this
in your documents." An answer that cites nothing is labelled "Not from your
documents" and shows no sources. When a claim loses every citation it had
but other claims keep theirs, the claim stays and the answer carries the
note "Parts of this answer aren't linked to a source." A sentence the model
never cited gets no note.

Thirteen questions were asked of gpt-oss-120b through the app's
`prepare()` (13 calls). All 12 answers were correct and 12 of 13 carried
citations (the 13th was the model's own decline). None of the 20
citations relied on a heading. Read against the chunks by hand, the word
matcher of the previous release would have declined 1 to 3 of those 12
correct answers under this rule. The matcher now normalises Unicode
hyphens (gpt-oss writes phone numbers with U+2011), checks a citation run on a
line of its own against the list above it, ignores one- and two-letter
codes such as "GV", and does not end a claim at "B.S.". Replayed, all 12
answers keep a supported citation.

A document title (a heading that opens every chunk's heading path, such as
a resume's name) now stays in the path only of the chunks whose text
prints it. Measured with this harness after re-ingesting the eval
account's documents (zero LLM calls):

| Title in the heading path | Answer in context | s19 name chunk (dense / context) | s20 (dense / context) | Citations | Off-topic declined by the gate |
|---|---|---|---|---|---|
| Every chunk (before) | 14 of 14 | 4 / 6 | 3 / 1 | 112 of 112 | 5 of 8 |
| Only chunks that print it (this change) | 14 of 14 | 4 / 6 | 6 / 1 | 112 of 112 | 6 of 8 |
| Every chunk, moved to the end of the embedding text | 14 of 14 | 6 / 6 | 3 / 1 | 112 of 112 | 6 of 8 |

Neither change moves an answer in or out of the context. The chosen one
also takes the title out of the prompt's header lines, so the model reads
a resume's name only from the chunk that states it. The lowest answerable
cosine went from 0.503 to 0.500 (s19); "Summarize the plot of Hamlet"
against the resume fell from 0.488 to 0.463 and is now declined by the
gate.

## Small workspaces

A workspace is small when dense search returns every chunk the user can
search there (fewer hits than `RETRIEVAL_CANDIDATE_K`, 20) and they all
fit the prompt's context budget (`RAG_CONTEXT_TOKENS`, 2,500, counted the
way the prompt builder counts). Then each document's first chunk always
joins the prompt, after the reranker's top 8, and the answer gate never
declines: the LLM decides. Both resumes qualify (8 chunks are 834 tokens,
17 chunks are 1,058); the samples workspace (125 chunks) does not.

Measured with zero LLM calls, 56 questions, the same ingested documents:

| | Master | This change |
|---|---|---|
| False declines | 0 | 0 |
| Must-decline sent to the summary route | 0 | 0 |
| Must-decline sent to the LLM, which decides | 11 | 15 |
| Answer in context (16 with `answer`) | 15 (v09 misses the name) | 16 |
| Shown citations on a chunk that states the answer | 128 of 128 | 130 of 130 |
| Resume questions whose prompt has every first chunk | 20 of 22 (4 declined by the gate) | 26 of 26 |
| Regressions | 1 (v09) | 0 |

The four must-decline questions that now reach the LLM are the off-topic
ones against the first resume (o01, o03, o06, o07), which the gate used to
decline. The first-chunk rule adds about 56 estimated prompt tokens, and
only to the two questions it changes (s21, v09). The four questions now
sent to the LLM cost a full prompt each (about 1,090 estimated tokens),
where the gate used to cost nothing. Over the 26 small-workspace questions
that is 172 estimated tokens per question.

The name question with the cosine cut-off raised above its score (0.51,
like a one-page resume whose name question scores just under 0.48):

| | `rag_gate` line for "What is the name of the candidate?" |
|---|---|
| Master | `cosine_top=0.5004 cosine_threshold=0.51 reranker_top=0.0 zone=decline` |
| This change | `cosine_top=0.5004 cosine_threshold=0.51 reranker_top=0.0 small_workspace=True zone=llm` |

Then 7 questions were asked of gpt-oss-120b with this change's prompts
(7 calls, one sample each at temperature 0.1):

| Question | Answer after the citation rule |
|---|---|
| s19 "What is the name of the candidate?" | "Rosalind Kettleborne", citing the page 1 chunk that states it (already in the top 8, so the prompt is master's) |
| v09 "Tell me about the candidate." | A profile with 10 citations, all on chunks that state their claims; it names her, and the name is only in the first chunk, which this change added |
| s21 "What talks has the candidate given?" | Both talks, citing the Talks chunk |
| o01, o03, o06, o07 (off-topic, first resume) | "I cannot find this in your documents.", no sources |

Groq counts a prompt as its o200k tokens plus 74 per call; the app's
cl100k estimate is 47 to 74 tokens low. Measured extra cost: 55 prompt
tokens where a first chunk is added (v09, s21), and about 1,170 tokens for
each off-topic question that now reaches the model. Over the 26
small-workspace questions that is 184 extra Groq tokens per question,
98% of it the four off-topic calls.

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

## The answer gate

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
