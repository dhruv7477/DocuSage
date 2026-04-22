# DocuSage

A multi-tenant, AI-powered private knowledge assistant. Upload your documents and get accurate, cited answers backed entirely by your own data — no hallucinations from outside sources.

---

## What it does

- Upload PDFs, DOCX, TXT, Markdown, and HTML files
- Documents are chunked, embedded, and stored in a vector database
- Ask questions in natural language — answers are sourced exclusively from your documents
- Full multi-tenant isolation: each workspace is a private silo
- Real-time streaming responses via SSE

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Zustand |
| Backend | FastAPI, SQLAlchemy (async), Alembic |
| Database | PostgreSQL 16 |
| Vector DB | Qdrant |
| Cache / Queue | Redis, Celery |
| Object Storage | MinIO (local) or AWS S3 |
| LLM | Ollama (local) · OpenAI · Anthropic · vLLM |
| Embeddings | BAAI/bge-m3 (local) · OpenAI · Ollama |

---

## Prerequisites

Install these before anything else.

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Docker + Docker Compose | Latest | [docker.com](https://www.docker.com/get-started/) |
| Git | Any | [git-scm.com](https://git-scm.com/) |

> **Windows:** Use PowerShell or Git Bash. WSL2 is recommended for a smoother experience.

> **Mac:** Install [Homebrew](https://brew.sh/) first, then `brew install python node`.

> **Linux:** Use your distro's package manager (`apt`, `dnf`, `pacman`, etc.).

---

## Option A — Docker (recommended, works on any OS)

This starts every service with one command. No manual database or service setup required.

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd DocuSage
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# Required — generate a random 32+ character string
SECRET_KEY=change-me-to-a-random-32-char-string-minimum

# Required — pick a password
POSTGRES_PASSWORD=your_db_password

# Required — choose your LLM provider
LLM_PROVIDER=ollama          # ollama | openai | anthropic | vllm
LLM_MODEL=llama3.2:3b        # model name for the provider above

# If using OpenAI:
# OPENAI_API_KEY=sk-...
# LLM_MODEL=gpt-4o

# If using Anthropic:
# ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL=claude-3-5-sonnet-20241022

# Embeddings — bge runs locally, no API key needed
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-m3

# App
APP_ENV=development
APP_DEBUG=true
POSTGRES_DB=docusage
POSTGRES_USER=docusage_user
```

> **Tip:** Generate a secret key with `python -c "import secrets; print(secrets.token_hex(32))"`

### 3. Start all services

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts: API (`:8000`), PostgreSQL (`:5432`), Qdrant (`:6333`), Redis (`:6379`), MinIO (`:9000`), Ollama (`:11434`).

The first run downloads Docker images and the embedding model (~2 GB). This takes several minutes. Subsequent starts are fast.

### 4. Run database migrations

In a second terminal, once the API container is healthy:

```bash
docker compose -f docker/docker-compose.yml exec api alembic upgrade head
```

### 5. Pull an LLM (Ollama only)

If you chose `LLM_PROVIDER=ollama`:

```bash
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3.2:3b
```

Replace `llama3.2:3b` with any model from [ollama.com/library](https://ollama.com/library). Larger models give better answers but need more RAM.

| Model | RAM needed | Quality |
|---|---|---|
| `llama3.2:3b` | ~4 GB | Good for testing |
| `llama3.1:8b` | ~8 GB | Recommended |
| `llama3.3:70b` | ~48 GB | Best quality |

### 6. Start the frontend

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Option B — Local development (no Docker for the app)

Use this for faster Python iteration. You still need Docker for the infrastructure services.

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd DocuSage
```

### 2. Start infrastructure via Docker

```bash
docker compose -f docker/docker-compose.yml up postgres redis qdrant minio -d
```

If using Ollama locally (outside Docker), [install it](https://ollama.com/) and run:

```bash
ollama serve          # keep this running in a terminal
ollama pull llama3.2:3b
```

### 3. Create a Python virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
python -m venv .venv
source .venv/Scripts/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> `torch` and `sentence-transformers` total ~2–4 GB. This step takes several minutes on first run.

> **GPU acceleration (optional):** If you have an NVIDIA GPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Then set `EMBEDDING_DEVICE=cuda` in your `.env`.

> **Apple Silicon (M1/M2/M3):** PyTorch runs natively. Set `EMBEDDING_DEVICE=mps` for GPU acceleration.

### 5. Create your environment file

```bash
cp .env.example .env
```

Fill in `.env` with the same values as Option A, plus these local connection overrides:

```env
POSTGRES_HOST=localhost
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
QDRANT_URL=http://localhost:6333
MINIO_ENDPOINT=localhost:9000
OLLAMA_BASE_URL=http://localhost:11434
```

### 6. Run database migrations

```bash
alembic upgrade head
```

### 7. Start the backend API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

On first startup the embedding model (~2 GB) is downloaded and loaded. Wait for `embedding_warmup_done` in the logs before making requests.

### 8. Start the Celery worker

Open a second terminal with the venv activated:

```bash
celery -A src.workers.celery_app worker --loglevel=info --concurrency=2
```

### 9. Start the frontend

Open a third terminal:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Using the app

1. **Register** an account at [http://localhost:3000](http://localhost:3000)
2. A workspace is created automatically and selected for you
3. Click **+ New Chat** in the sidebar — the Documents panel opens automatically
4. **Upload files** via drag-and-drop or click to browse (PDF · DOCX · TXT · MD · HTML)
5. Wait for the status badge to show **Ready** — the indexing banner will disappear
6. Type your question in the chat input and get cited answers from your documents

You can also open the Documents panel at any time using the floating **Documents** button in the bottom-right of the screen.

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | **Yes** | — | JWT signing key, min 32 chars |
| `POSTGRES_PASSWORD` | **Yes** | — | PostgreSQL password |
| `POSTGRES_DB` | No | `docusage` | Database name |
| `POSTGRES_USER` | No | `docusage_user` | Database user |
| `POSTGRES_HOST` | No | `localhost` | Database host |
| `LLM_PROVIDER` | No | `ollama` | `ollama` · `openai` · `anthropic` · `vllm` |
| `LLM_MODEL` | No | `llama3.3:70b` | Model name for the chosen provider |
| `OPENAI_API_KEY` | If OpenAI | — | `sk-...` |
| `ANTHROPIC_API_KEY` | If Anthropic | — | `sk-ant-...` |
| `EMBEDDING_PROVIDER` | No | `bge` | `bge` · `openai` · `ollama` |
| `EMBEDDING_MODEL` | No | `BAAI/bge-m3` | Embedding model name |
| `EMBEDDING_DEVICE` | No | `cpu` | `cpu` · `cuda` · `mps` |
| `STORAGE_BACKEND` | No | `minio` | `minio` · `s3` |
| `MINIO_ENDPOINT` | No | `localhost:9000` | MinIO address |
| `MINIO_ACCESS_KEY` | No | `minioadmin` | MinIO root user |
| `MINIO_SECRET_KEY` | No | `minioadmin` | MinIO root password |
| `GOOGLE_CLIENT_ID` | No | — | Enable Google OAuth sign-in |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant address |
| `APP_ENV` | No | `development` | `development` · `staging` · `production` |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |
| `RATE_LIMIT_CHAT_PER_MINUTE` | No | `20` | Per-user chat rate limit |

---

## Development commands

```bash
# Backend linting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy

# Unit tests (no services needed)
pytest tests/unit/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v

# All tests with coverage report
pytest tests/ -v --cov

# Frontend dev server
npm run dev

# Frontend production build
npm run build

# Preview production build locally
npm run preview
```

---

## Project structure

```
DocuSage/
├── src/                      # Backend (Python / FastAPI)
│   ├── api/v1/               # Route handlers
│   ├── services/             # Business logic
│   ├── models/               # SQLAlchemy ORM models
│   ├── schemas/              # Pydantic request/response models
│   ├── core/                 # JWT, audit logging, rate limiting, exceptions
│   ├── ingestion/            # Document parsing and chunking pipeline
│   ├── providers/            # Pluggable LLM and embedding backends
│   ├── retrieval/            # Query rewriting, reranking, BM25 hybrid search
│   ├── vector_store/         # Qdrant client
│   ├── workers/              # Celery document processing tasks
│   ├── db/                   # Session factory and Alembic migrations
│   └── configs/              # Settings (single source of truth)
├── src/                      # Frontend (React / TypeScript)
│   ├── components/           # UI components
│   ├── pages/                # App shell and Auth page
│   ├── store/                # Zustand state stores
│   ├── hooks/                # SSE streaming hook
│   └── api/                  # HTTP client and API definitions
├── tests/
│   ├── unit/                 # Pure unit tests, no I/O
│   └── integration/          # FastAPI TestClient tests
├── docker/
│   ├── docker-compose.yml    # Full local stack
│   └── Dockerfile
├── .env.example              # Environment variable template
└── requirements.txt          # Python dependencies
```

---

## Troubleshooting

**`ECONNREFUSED` errors in the browser console**
The backend is not running. Start uvicorn (Option B) or `docker compose up` (Option A).

**`relation "users" does not exist`**
Database migrations have not been run. Execute `alembic upgrade head`.

**Embedding model downloads on first startup**
This is expected. The ~2 GB BGE-M3 model is downloaded once and cached. Subsequent starts load from the local cache in ~30 seconds.

**Documents stuck on "Indexing"**
The Celery worker is not running. In a separate terminal (with venv active): `celery -A src.workers.celery_app worker --loglevel=info`

**`No relevant documents found` when chatting**
Either no documents have been uploaded, or they are still indexing (status not yet "Ready"). Check the Documents panel.

**Port already in use**
One of the required ports (5432, 6333, 6379, 8000, 9000) is taken by another process. Either stop that process or change the port mapping in `docker/docker-compose.yml`.

**Out of memory during embedding**
The BGE-M3 model uses ~2 GB of RAM. If you're on a constrained machine, switch to a lighter model: set `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5` and `EMBEDDING_DIMENSION=768` in `.env`.
