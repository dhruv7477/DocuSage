# ════════════════════════════════════════════════════════════════
#  DocuSage — Developer Makefile
# ════════════════════════════════════════════════════════════════
#
# Common targets to run the local stack the same way every time.
# The previous loose "follow steps 1..9 in the README" workflow was
# the source of most "I ran the API but Redis wasn't up" reports.

COMPOSE       ?= docker compose --env-file .env -f docker/docker-compose.yml
# Dev answer model: 3B on this ARM/CPU box (fits 16 GB, ~2x faster than 7B).
# Override: make LLM_MODEL=... (the public demo uses a hosted model instead).
LLM_MODEL     ?= qwen2.5:3b-instruct
ROUTER_MODEL  ?= qwen2.5:1.5b-instruct

# Celery's default `prefork` pool does not work on Windows — workers
# come up but never actually pick up tasks. Detect Windows and pick a
# pool that works. `solo` runs one task at a time on the main thread —
# fine for development; on Linux/macOS we use `prefork` with concurrency 2.
ifeq ($(OS),Windows_NT)
  CELERY_POOL_FLAGS ?= --pool=solo
else
  CELERY_POOL_FLAGS ?= --pool=prefork --concurrency=2
endif

.DEFAULT_GOAL := help


# ── Help ──────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "DocuSage local-dev targets"
	@echo ""
	@echo "  Full Docker stack (everything in containers)"
	@echo "    make stack-up         Build + start the full stack (api, worker, deps)"
	@echo "    make stack-down       Stop and remove containers (volumes preserved)"
	@echo "    make stack-logs       Tail logs from every service"
	@echo "    make stack-reset      Stop + remove volumes (WIPES data)"
	@echo ""
	@echo "  Local-dev workflow (deps in Docker, app on host)"
	@echo "    make dev-up           Start deps (postgres/redis/qdrant/seaweedfs/ollama),"
	@echo "                          pull the LLM model"
	@echo "    make dev-up-native    Like dev-up but Ollama runs NATIVELY (ARM64/Snapdragon)"
	@echo "    make dev              Run API + Celery worker + frontend in ONE terminal"
	@echo "    make dev-down         Stop dep containers"
	@echo "    make dev-status       Show whether each dep is reachable"
	@echo "    make worker           Run the Celery worker alone (Windows-safe pool)"
	@echo "    make api              Run the FastAPI dev server alone"
	@echo ""
	@echo "  Database"
	@echo "    make migrate          Run Alembic migrations against the local DB"
	@echo "    make migrate-docker   Run migrations inside the api container"
	@echo ""
	@echo "  Quality"
	@echo "    make lint             Ruff + format-check + mypy"
	@echo "    make test             Unit + integration tests"
	@echo "    make eval             RAG evaluation harness (Phase 5)"


# ── Full Docker stack ─────────────────────────────────────────────
# The bootstrap container (ollama-bootstrap) is removed
# before every `up` so compose re-runs them. Without this, compose's
# `service_completed_successfully` policy skips the bootstrap on the
# second `up` — which is the trap that left an outdated LLM_MODEL stuck
# in the volume after `.env` changed. The bootstrap scripts themselves
# are idempotent (no-op when the model already exists), so the
# cost of re-running on every boot is effectively zero.
.PHONY: stack-up stack-down stack-logs stack-reset
stack-up:
	-$(COMPOSE) rm -fs ollama-bootstrap 2>/dev/null
	$(COMPOSE) up --build -d
	@echo "Stack starting. The model-pull bootstrap"
	@echo "may take a few minutes on first run. Tail with: make stack-logs"

stack-down:
	$(COMPOSE) down

stack-logs:
	$(COMPOSE) logs -f

stack-reset:
	$(COMPOSE) down -v


# ── Local dev (deps in Docker, app on host) ───────────────────────
.PHONY: dev-up dev-down dev-status
dev-up:
	@echo "Stopping containerized api/worker/postgres. api+worker double-bind"
	@echo "port 8000 and steal Celery tasks when run alongside the host app."
	@echo "Postgres runs NATIVELY on this machine (service postgresql-x64-18,"
	@echo "where the dev data lives) — the container postgres would double-bind"
	@echo "5432 and split reads/writes across two different databases."
	-$(COMPOSE) stop api worker postgres 2>/dev/null
	$(COMPOSE) up -d redis qdrant seaweedfs ollama
	@echo "Re-running the model bootstrap so it picks up any .env changes..."
	-$(COMPOSE) rm -fs ollama-bootstrap 2>/dev/null
	$(COMPOSE) up ollama-bootstrap
	@echo ""
	@echo "Deps are up. Now:"
	@echo "  1)  make migrate     # one-shot, run Alembic migrations"
	@echo "  2)  make dev         # API (:8000) + Celery worker + frontend (:3000)"
	@echo "                       # all in this one terminal — Ctrl+C stops all"

dev-down:
	$(COMPOSE) stop redis qdrant seaweedfs ollama \
	  ollama-bootstrap


# ── Native Ollama (Windows ARM64 / Snapdragon dev) ────────────────
# This machine is ARM64 (Snapdragon X). Running Ollama NATIVELY instead
# of in the WSL2 Docker container uses the Oryon CPU directly and avoids
# the WSL2 memory caps that OOM-kill the model and wedge the engine.
# One-time: install native Ollama (Windows ARM64) from
# https://ollama.com/download — it runs a background service on :11434.
# Then use `make dev-up-native` instead of `make dev-up`.
.PHONY: dev-up-native ollama-pull
dev-up-native:
	@echo "Native-Ollama dev: deps in Docker, Ollama native on :11434."
	-$(COMPOSE) stop api worker postgres ollama ollama-bootstrap 2>/dev/null
	$(COMPOSE) up -d redis qdrant seaweedfs
	@$(MAKE) ollama-pull
	@echo ""
	@echo "Deps up + models pulled. Make sure native Ollama is running"
	@echo "(curl http://localhost:11434/api/tags), then: make dev"

# Pull answer + router models into the NATIVE Ollama via its HTTP API.
# Uses curl (always on PATH via System32) instead of the `ollama` CLI,
# which GNU Make on Windows can't resolve (CreateProcess doesn't add .exe
# / search PATH like a shell). Idempotent: already-present models return
# fast. Requires native Ollama running on :11434.
ollama-pull:
	@echo "Pulling models into native Ollama (:11434): $(LLM_MODEL), $(ROUTER_MODEL)"
	@curl -fsS http://localhost:11434/api/pull -d '{"name":"$(LLM_MODEL)"}' >/dev/null || (echo "ERROR: native Ollama not reachable on :11434 — is it installed and running?" && exit 1)
	@curl -fsS http://localhost:11434/api/pull -d '{"name":"$(ROUTER_MODEL)"}' >/dev/null


# ── App processes ─────────────────────────────────────────────────
# `make dev` = API + worker + frontend in one terminal via concurrently
# (color-prefixed logs, Ctrl+C stops all three). Run `make dev-up` once
# first so the Docker deps are alive.
#
# NOTE: `make dev` runs uvicorn WITHOUT --reload. On Windows, uvicorn's
# reloader broadcasts CTRL_C_EVENT to the whole console group when it
# restarts its worker, which kills the npm/concurrently wrapper (the
# API dies with KeyboardInterrupt right after a .py file changes).
# For backend hot-reload, run `make api` in its OWN terminal instead.
.PHONY: dev api worker
dev:
	npm run dev:full

# Port 8010 on purpose: the Docker stack's api container publishes 8000,
# and Windows lets both bind simultaneously (requests then round-robin
# between a healthy and a dead server). Separate ports = no collision.
api:
	uvicorn src.main:app --host 127.0.0.1 --port 8010 --reload

# Without this worker the document-upload pipeline is dead: API saves
# the row + blob, dispatches a Celery task, then nothing pulls it off
# the Redis queue. Documents stay PENDING forever. Always run this in
# a second terminal alongside `make api`.
worker:
	celery -A src.workers.celery_app worker --loglevel=info -Q ingestion,default,celery $(CELERY_POOL_FLAGS)

dev-status:
	@echo "PostgreSQL (native):" && "C:\Program Files\PostgreSQL\18\bin\pg_isready.exe" -h localhost -U docusage_user -d docusage || true
	@echo "Redis:    " && $(COMPOSE) exec -T redis redis-cli ping || true
	@echo "Qdrant:   " && curl -sf http://localhost:6333/healthz && echo " ok" || true
	@echo "S3:       " && curl -sf http://localhost:9000/healthz && echo " ok" || true
	@echo "Ollama:   " && curl -sf http://localhost:11434/api/tags >/dev/null && echo "ok" || echo "down"


# ── Database ──────────────────────────────────────────────────────
.PHONY: migrate migrate-docker
migrate:
	alembic upgrade head

migrate-docker:
	$(COMPOSE) exec api alembic upgrade head


# ── Quality ───────────────────────────────────────────────────────
.PHONY: lint test eval
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy

test:
	pytest tests/ -v --cov

eval:
	python -m tests.eval.run_eval
