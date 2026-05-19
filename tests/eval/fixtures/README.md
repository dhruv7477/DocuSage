# Evaluation fixtures

Drop the documents you want the harness to ingest into this folder.
The harness uploads every file (matching `*.pdf`, `*.docx`, `*.txt`,
`*.md`, `*.markdown`, `*.html`) into a fresh workspace named
`eval-{timestamp}` and waits for each to reach `status=completed`.

Then create your question set: copy `qa.example.jsonl` to `qa.jsonl`
and edit. Each line is one record:

```jsonl
{"question": "...", "expected_keywords": ["..."], "expected_doc": "filename.pdf", "expected_page": 3}
```

* `expected_doc` may be `null` when the question is intentionally
  unanswerable from the corpus — the harness checks the answer
  contains the abstention phrase ("I cannot find this in your
  documents.") and counts that as success.
* `expected_page` may be `null` to skip the page-level citation
  accuracy check for that question.
