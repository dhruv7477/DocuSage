# DocuSage: Technical Deep Dive

A reference for engineers and technical stakeholders. It covers the architecture as built, then a forward-looking section on orchestrating the same workflow with **LangGraph**, and why LangGraph is the right tool over plain LangChain for this kind of system.

---

## 1. System at a Glance

DocuSage is a multi-tenant, retrieval-augmented-generation (RAG) platform: users upload documents and get cited, document-grounded answers through a chat interface.

**Stack**
- **Backend:** FastAPI + SQLAlchemy (async), entry point `src/main.py:create_app()`.
- **Frontend:** React 18 + TypeScript + Vite, SSE-driven chat UI.
- **Datastores:** PostgreSQL (relational), Qdrant (vectors), Redis (Celery broker, revoked-token list, rate-limit counters), S3-compatible object storage for the original files (SeaweedFS in the compose files, or Amazon S3).
- **Async work:** a Celery worker runs the ingestion pipeline.
- **Models:** pluggable LLM providers (Ollama, OpenAI or any OpenAI-compatible API such as Groq, vLLM) and embedding providers (BGE in-process, OpenAI), selected at runtime through config factories.

**Layering (backend)**
- `src/api/v1/`: thin route handlers (`auth`, `chat`, `documents`, `workspaces`, `admin`, `eval`, `demo`).
- `src/services/`: all business logic (orchestration, RAG, summarization, validation, prompt building, quotas).
- `src/retrieval/`: query understanding, hybrid search, reranking.
- `src/ingestion/`: parsers, chunker, embedding, vector insert.
- `src/providers/`, `src/vector_store/`: provider and store abstractions behind factories.
- `src/core/`: exceptions, security (JWT + bcrypt), structured logging, rate limiting, audit.

---

## 2. Document Ingestion Pipeline

Upload is decoupled from indexing so the API stays responsive:

```
Upload → object storage (raw file) + Document row (PENDING)
       → Celery task: parse → chunk → embed → Qdrant upsert
       → status transitions → COMPLETED
```

- **Parsers** handle PDF, DOCX (and DOC), TXT, Markdown and HTML.
- **Chunker** splits text into overlapping passages with heading context preserved (`heading_path`), which later powers precise citations.
- **Embedding** turns each chunk into a dense vector (default `BAAI/bge-small-en-v1.5`, 384 dimensions).
- **Qdrant upsert** writes vectors plus a payload (`document_id`, `user_id`, `page_number`, `text`, `heading_path`, character offsets) and a **full-text payload index** that enables keyword search alongside vector search.
- Progress and failures are persisted on the `Document` row and surfaced live in the UI. A periodic sweep marks documents stuck in processing as failed, so none stays pending forever.

---

## 3. Retrieval Pipeline (`src/retrieval/`)

This is the precision core of the RAG path:

1. **Query understanding** (`query_understanding.py`), two cheap, gated steps:
   - *Standalone rewriting:* follow-ups like "and what about section 2?" are rewritten into self-contained queries using recent chat history. Gated by a regex for anaphora **and** the presence of prior turns, so first turns don't spend an LLM call.
   - *Optional HyDE:* generate a hypothetical answer and embed that instead; a recall booster, off by default because it adds latency on local models.
2. **Hybrid retrieval** (`hybrid.py`): runs **dense** vector search and **keyword** full-text search concurrently, then fuses them with **Reciprocal Rank Fusion (RRF)**. This rescues exact-keyword matches the embedder might miss. All reads are filtered by `user_id` and workspace.
3. **Reranking + abstention** (`reranker.py`): a **cross-encoder** (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, configurable, for example to `bge-reranker-v2-m3`) reads `(query, candidate)` pairs together to model fine-grained relevance such as negation and numbers. Scores are **sigmoid-normalized to [0,1]** so a single `abstention_threshold` governs whether retrieval is confident enough to answer at all. Below the threshold the system declines rather than hallucinates.

Flow: `hybrid retrieve (about top 50) → cross-encoder rerank → top-k → grounded prompt`.

---

## 4. Orchestration (as built)

The chat path is composed, not monolithic. It implements the **routing workflow** from Anthropic's *Building Effective Agents*: classify the input, then dispatch to a specialized path.

**`ChatOrchestrator.prepare()`** is the "prepare" half of a turn:

- **`QuestionAnalyzer`** (the router) classifies intent into `off_topic | doc_summary | rag_query`. Rules come first and cost nothing: greetings get a fixed friendly reply, and summary requests ("summarize report.pdf") go to the summarizer, narrowed to any file named in the question. Everything else is `rag_query`. An optional small JSON-mode LLM classifier (`ROUTER_USE_LLM`, off by default) handles the remaining cases; it is **fail-open**, so any parse error, unknown intent or LLM failure falls back to `rag_query` and a broken router never blocks a real question.
- Dispatch:
  - **`off_topic`**: a short reply, pseudo-streamed in slices. No retrieval.
  - **`doc_summary`**: `DocSummaryService`, map-reduce over a document's chunks. Oversized documents are split into parts, each **streamed** (so the read timeout only has to cover the time to the first token, not the whole reply), then reduced into a final streamed executive summary. `DOC_SUMMARY_ENABLED=false` sends summary requests down the `rag_query` path with a warning, for deployments whose LLM quota cannot afford several calls per question.
  - **`rag_query`**: `RAGService`, query understanding → hybrid retrieve → rerank → `prompt_builder` assembles the grounded prompt → token stream. When reranking finds nothing relevant enough, the turn is declined with a fixed reply instead of generating.

The orchestrator returns a **`StreamPlan`**: a token iterator plus everything needed after generation (sources in context order, confidence, the abstained flag, and whether `[n]` citations are legitimate).

**Transport (`/chat/{id}/stream`, SSE):** a producer coroutine runs `prepare()` and feeds a queue; the endpoint emits `thinking`, `route`, `token`, `sources`, `warning` and `done` events. An 8-second **keepalive** heartbeat keeps the connection alive during long local-model work. After the last token, `answer_validator` runs **programmatically** (no second LLM pass) to confirm citations are valid and the answer respected the abstention contract; messages are then persisted to PostgreSQL.

**Grounding and proof:** citations map to context-ordered sources. The `documents/{id}/page-image` endpoint renders the exact PDF page with the cited passage highlighted (PyMuPDF), optionally cropped to just the paragraph: the UI's "Check source" preview.

---

## 5. Multi-Tenancy, Security, Configuration

- **Isolation:** each workspace gets its own Qdrant collection (`{prefix}_ws_{workspace_id}`); every query *additionally* filters by `user_id`; object-storage paths are workspace-prefixed. Tests against a real Qdrant check that one tenant's search never returns another's chunks.
- **Auth:** JWT (PyJWT) + bcrypt; dependency guards (`get_current_user`, `require_admin`, `require_super_admin`); an audit log in `src/core/`.
- **Abuse controls for public deployments:** per-IP limits on sign-up and sign-in, per-user limits on questions, uploads and page images, per-user quotas (documents, upload size, questions per day), and an optional site-wide daily question budget. Quotas default to off for self-hosted installs.
- **Config:** single source of truth in `src/configs/config.py` (Pydantic `BaseSettings`, `@lru_cache`). Models are swappable per role (separate settings for **chat**, the **router** and **summarization**), so cost and latency can be tuned without code changes.
- **Quality:** an `eval` API and `eval_service` measure RAG quality and track regressions. CI runs ruff, unit and integration tests (against real Postgres, Redis, Qdrant and SeaweedFS), the frontend tests and build, and the whole demo stack end to end. Separate workflows run dependency audits, CodeQL, a secret scan, container image scans and an OWASP ZAP scan on every pull request.

---

## 6. Reference Architecture: Orchestration with LangGraph

> **Note:** today the orchestration is a purpose-built async controller (`ChatOrchestrator` + a producer/queue SSE loop). It already behaves like a small state machine. This section describes how we would express the same workflow in **LangGraph**, and why we'd choose it over plain LangChain if and when the agent logic grows.

### The workflow as a graph
LangGraph models an application as a **`StateGraph`**: a typed shared state object flows through **nodes** (functions or agents) connected by **edges**, including **conditional** and **cyclic** edges.

```
              ┌─────────────┐
   START ───▶ │  router     │  (QuestionAnalyzer)
              └──────┬──────┘
        off_topic │  doc_summary │  rag_query
        ▼          ▼              ▼
   ┌─────────┐ ┌───────────┐ ┌──────────────┐
   │ decline │ │ summarize │ │ understand_q │
   └────┬────┘ │ (map→     │ └──────┬───────┘
        │      │  reduce)  │        ▼
        │      └─────┬─────┘   ┌──────────┐
        │            │         │ retrieve │ (hybrid + RRF)
        │            │         └────┬─────┘
        │            │              ▼
        │            │         ┌──────────┐   below threshold
        │            │         │ rerank   ├───────────────▶ decline
        │            │         └────┬─────┘
        │            │              ▼
        │            │         ┌──────────┐
        │            │         │ generate │
        │            │         └────┬─────┘
        └────────────┴──────────────┤
                                     ▼
                              ┌────────────┐   citations invalid →
                              │  validate  ├──────── (loop) ──────▶ regenerate / refine
                              └─────┬──────┘
                                    ▼
                                   END
```

**Shared state** (a typed `TypedDict` or Pydantic model) carries: `query`, `chat_history`, `intent`, `enriched_queries`, `candidates`, `reranked`, `confidence`, `answer`, `sources`, `abstained`. Each node reads and updates the parts it owns.

**Mapping to existing components**, almost one to one:
- `router` → `QuestionAnalyzer`
- `understand_q` / `retrieve` / `rerank` → `src/retrieval/*`
- `generate` → `RAGService` + `prompt_builder`
- `summarize` → `DocSummaryService` (LangGraph's `Send` API expresses the map step as fan-out and the reduce step as a join, a natural fit for map-reduce)
- `validate` → `answer_validator`, with a **conditional edge** that could loop back to `generate` for a refine attempt. Today the validator corrects the answer in place and does not retry.

**What LangGraph gives us for free:**
- **Conditional routing and cycles:** the router branch and a validate→regenerate retry loop are first-class edges, not hand-rolled `if/else` and ad-hoc loops.
- **Streaming of intermediate steps:** `graph.astream_events` emits node-level progress, mapping cleanly onto our SSE `thinking`/`token` events.
- **Checkpointing and durable state:** built-in checkpointers persist state per thread, enabling resumability, time-travel debugging and human-in-the-loop approvals.
- **Observability:** the graph is introspectable and integrates with LangSmith for tracing each node's inputs and outputs.

### Why LangGraph over LangChain (for orchestration)

LangChain and LangGraph are complementary, not competitors. LangGraph is built **by** the LangChain team and happily uses LangChain components (LLM wrappers, retrievers, tools) *inside* its nodes. The distinction is about **control flow**:

| Concern | LangChain (chains/LCEL) | LangGraph |
|---|---|---|
| Control flow | Linear pipelines / DAGs (`A → B → C`) | Explicit graph with **branches, loops, and conditionals** |
| Branching by intent | Possible (`RouterChain`) but awkward and rigid | Native **conditional edges** |
| Cyclic flows (retry/refine, agent loops) | Hard: chains are meant to terminate forward | First-class: designed for cycles |
| Shared, evolving state | Passed implicitly through the chain; hard to inspect | **Central typed state**, explicitly read and updated by each node |
| Pause / resume / human-in-loop | Not built in | **Checkpointers** make it native |
| Streaming partial progress | Final-output oriented | **Per-node event streaming** |
| Map-reduce fan-out | Manual | **`Send` API** for dynamic fan-out and fan-in |

**The bottom line for DocuSage:** our flow is *not* a straight pipeline. It **branches** on intent, could **loop** (validate → regenerate, or retrieve → re-query on low confidence), **fans out and in** for summarization, and needs **mid-flight streaming** of progress. Those are precisely the patterns LangChain's chain abstraction strains to express and LangGraph models directly. LangChain remains the right choice for the *building blocks* (LLM and embedding adapters, retrievers, output parsers); LangGraph is the right choice for the *orchestration* that wires those blocks into a stateful, conditional graph. Adopting it would replace our bespoke controller with a standardized, observable, resumable graph: less custom plumbing, and easier to extend with new steps (for example a "clarifying question" node or a "tool use" node) by adding nodes and edges.

---

## 7. Deployment and Operations
- **Local development:** Docker Compose brings up Postgres, Qdrant, Redis, SeaweedFS and Ollama; the API, the Celery worker and Vite run on the host.
- **Public demo:** one Compose file runs the whole stack on a single small VM (built and tested on linux/arm64), with Caddy as the only public entry point (automatic HTTPS, security headers) and a hosted LLM (Groq) through the OpenAI-compatible provider. See `deploy/oracle/README.md`.
- **Migrations:** Alembic (`alembic upgrade head`), run automatically when the API container starts.
- **CI:** lint → unit and integration tests → frontend → image builds for amd64 and arm64 → the demo stack end to end on arm64, plus the security workflows above. Images are built on the server from the repository; nothing is pushed to a registry.
- **Tuning levers:** model selection per role, retrieval top-k and abstention threshold, chunk size and overlap, summary budgets, rate limits and quotas, all through environment configuration.
