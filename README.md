# DocuSage

[![CI](https://github.com/dhruv7477/DocuSage/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/dhruv7477/DocuSage/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

DocuSage is a multi-user knowledge assistant over your own documents. Upload PDFs and other files, ask questions in plain language, and get streamed answers with numbered citations that open the cited page.

## What it does

- Upload PDF, DOCX, TXT, Markdown and HTML files. A background worker parses, chunks and embeds them, and shows each stage while it runs.
- Ask questions. A small router model decides whether a message is a question about your documents, a request to summarise one, or small talk. Questions go through hybrid retrieval (dense vectors plus full-text search, fused with reciprocal rank fusion) and a cross-encoder reranker. The answer is streamed with `[n]` citations, and when nothing relevant is found it says so instead of guessing.
- Click a citation to see the source page with the passage highlighted.
- Workspaces hold documents and chats. Every account's data is isolated in two layers: a separate vector collection per workspace, and an owner filter on every vector query. Members of a shared workspace do **not** see each other's documents.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Zustand, TanStack Query |
| API | FastAPI, async SQLAlchemy, Alembic, slowapi |
| Worker | Celery with Redis |
| Data | PostgreSQL 16, Qdrant, any S3-compatible store (SeaweedFS in the compose files, or Amazon S3) |
| Embeddings and reranking | `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`, on the CPU, in-process |
| LLM | Ollama (local), or any OpenAI-compatible API (OpenAI, Groq, a vLLM server) |
| Demo hosting | Docker Compose behind Caddy (automatic HTTPS), tested on linux/arm64 |

## Quickstart: the demo stack

This runs the whole app in Docker, as on the public demo: Caddy with HTTPS, the API, a worker, Postgres, Redis, Qdrant and SeaweedFS. Answers come from Groq, so you need a free Groq API key from https://console.groq.com/keys.

Requirements: Docker with the Compose plugin, about 6 GB of free disk, and a POSIX shell (Linux, macOS, or Git Bash or WSL on Windows).

```bash
git clone https://github.com/dhruv7477/DocuSage.git
cd DocuSage
DEMO_DOMAIN=localhost GROQ_API_KEY=gsk_your_key scripts/demo-env.sh
docker compose -f docker/docker-compose.demo.yml up -d --build
```

`scripts/demo-env.sh` writes `docker/.env` with random secrets for every service. The first start builds two images and downloads about 220 MB of models, which takes about 10 minutes. `docker compose -f docker/docker-compose.demo.yml ps` shows every service as running or healthy when it is ready.

Then open https://localhost and accept the browser's certificate warning (Caddy issues a local certificate for `localhost`). Sign up. The sign-up page lists the demo limits. A few minutes after the first start, every new account has a "Sample documents" workspace with two public-domain PDFs. Ask, for example, "When must an employer complete Section 2 of Form I-9?", or upload your own file.

To stop the stack, run `docker compose -f docker/docker-compose.demo.yml down`. Add `-v` to delete all data.

The demo stack is in demo mode: each account can keep 5 documents (25 MB in total) and ask 30 questions a day, every upload is deleted at 03:00 UTC, and new accounts get the sample documents. All of these are settings in `docker/.env` (see `docker/.env.demo.example`).

To put the demo on a server with a real domain, follow [deploy/README.md](deploy/README.md). It covers an Oracle Cloud Always Free Ampere VM, but the steps work on any Ubuntu server with Docker.

## Local development

For working on the code: the infrastructure runs in Docker, and the API, worker and frontend run on your machine with hot reload. You need Python 3.11 or 3.12, Node 22 and Docker.

```bash
cp .env.example .env
docker compose --env-file .env -f docker/docker-compose.yml up -d postgres redis qdrant seaweedfs ollama
docker compose --env-file .env -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:3b-instruct
docker compose --env-file .env -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:1.5b-instruct

python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu   # CPU build, no CUDA download
pip install -r requirements-dev.txt  # pinned by constraints.txt
alembic upgrade head

npm ci
npm run dev:full                     # API on :8010, Celery worker, Vite on :3000
```

Open http://localhost:3000. The `.env` from `.env.example` works as is with those containers. `npm run dev:full` starts the API with `uvicorn src.main:app --port 8010`, the worker with `celery -A src.workers.celery_app worker --pool=solo` and the Vite dev server, which proxies `/api` to the API. The `Makefile` has the same steps as targets (`make dev-up`, `make migrate`, `make dev`).

To use Groq or OpenAI instead of a local Ollama, set the `LLM_PROVIDER=openai` block in `.env` (it is in the file, commented out).

## Configuration

All settings live in `src/configs/config.py` and come from environment variables or `.env`.

| Setting | Values |
|---|---|
| `LLM_PROVIDER` | `ollama` (default, `OLLAMA_BASE_URL`); `openai` with `OPENAI_API_KEY`, `OPENAI_MODEL` and optionally `OPENAI_BASE_URL` for any OpenAI-compatible API, such as `https://api.groq.com/openai/v1`; `vllm` with `VLLM_BASE_URL` |
| `LLM_MODEL`, `ROUTER_MODEL`, `SUMMARY_MODEL` | Answer model, question router model, and summary model (defaults to the router model). With `openai`, the answer model is `OPENAI_MODEL`. |
| `EMBEDDING_PROVIDER` | `bge` (default, local CPU) or `openai` |
| `RERANKER` | `cross_encoder` (default), `llm` or `none` |
| `STORAGE_BACKEND` | `s3_compatible` (default): any S3-compatible server, configured with `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` and `S3_SECURE` (SeaweedFS in the compose files); or `s3` for Amazon S3 with the `AWS_*` settings and `S3_BUCKET`. The old `minio` value and `MINIO_*` names still work for now and log a warning at startup. |
| `RATE_LIMIT_*` | Per-IP limits on sign-up and sign-in, per-user limits on questions, uploads, chats and page images. Counters live in memory, or in Redis with `RATE_LIMIT_STORAGE_URI`. |
| `QUOTA_*` | Per-user limits on documents, upload size and questions per day. 0 (the default) means unlimited. |
| `DEMO_GLOBAL_QUESTIONS_PER_DAY` | Questions per UTC day for all users together, to stay inside a hosted LLM's daily token limit. 0 (the default) means unlimited. |
| `DOC_SUMMARY_ENABLED` | Whole-document summaries (default on). When off, summary requests get a normal cited answer. |
| `LLM_REASONING_EFFORT` | `low`, `medium` or `high` for reasoning models such as gpt-oss on the `openai` provider. Unset by default. |
| `DEMO_MODE` | Sample documents for new accounts, a nightly wipe of uploads, and the notice on the sign-up page |

## Privacy

- Uploaded files are stored in your S3-compatible store, and chunks of their text are stored in Qdrant.
- With a hosted LLM (Groq or OpenAI), each question and the matching document excerpts are sent to that provider to write the answer. With Ollama, nothing leaves your server.
- The chat stream sends the question in the URL. The demo's Caddy configuration keeps no access logs for that reason.

## Tests

```bash
ruff check src/ tests/ scripts/ && ruff format --check src/ tests/ scripts/
pytest tests/unit                 # no services needed
pytest tests/integration          # tests that need Postgres, Redis, Qdrant or SeaweedFS skip when those are not reachable
npm test && npm run typecheck
```

CI (`.github/workflows/ci.yml`) runs all of these with real Postgres, Redis, Qdrant and SeaweedFS containers. It builds the images for amd64 and arm64, runs the demo stack end to end on arm64 with a mock LLM (`tests/e2e/`), and runs the VM setup script on a fresh runner.

Security checks run on every pull request and weekly: `.github/workflows/security.yml` (dependency review, pip-audit, npm audit, gitleaks, Trivy on both images, and an OWASP ZAP baseline and API scan against a disposable demo stack) and `.github/workflows/codeql.yml`. Results go to the repository's Security tab. To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Project layout

```
src/                 Backend (Python) and frontend (TypeScript) share this folder
  api/v1/            FastAPI routes
  services/          Business logic: documents, chats, quotas, routing, RAG, summaries
  retrieval/         Query rewriting, hybrid search, reranking
  ingestion/         Parsers, chunker, ingestion pipeline
  providers/         LLM and embedding providers
  vector_store/      Qdrant
  workers/           Celery app and tasks
  demo/              Sample documents and nightly cleanup for demo mode
  components/, pages/, hooks/, store/, api/   React frontend
docker/              Dockerfiles, compose files, Caddy and SeaweedFS config
deploy/              VM setup, backups and deployment guide
demo/samples/        Public-domain sample PDFs
docs/                Case study and technical deep dive
tests/               unit/, integration/, e2e/
```

For more depth: [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md) walks through ingestion, retrieval and orchestration, and [docs/CASE_STUDY.md](docs/CASE_STUDY.md) is the non-technical overview. [docs/DocuSage_Report.html](docs/DocuSage_Report.html) has both as one printable page.

## License

Apache License 2.0. See [LICENSE](LICENSE).
