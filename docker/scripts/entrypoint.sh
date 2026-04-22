#!/bin/sh
# docker/scripts/entrypoint.sh
# ─────────────────────────────
# Container entrypoint for both API and worker services.
#
# Responsibilities:
#   1. Wait for PostgreSQL to be ready (retry with backoff)
#   2. Run Alembic migrations (API service only, skipped for worker)
#   3. Exec the command passed to docker run / CMD
#
# Environment variables:
#   SKIP_MIGRATIONS=true   Skip migration step (for worker/beat containers)
#   POSTGRES_HOST          Used for pg_isready check
#   POSTGRES_USER          Used for pg_isready check
#   POSTGRES_DB            Used for pg_isready check

set -e

# ── Colours ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { printf "${GREEN}[entrypoint]${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}[entrypoint]${NC} %s\n" "$1"; }
err() { printf "${RED}[entrypoint]${NC} %s\n" "$1" >&2; }

# ── Wait for PostgreSQL ───────────────────────────────────────────
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_USER="${POSTGRES_USER:-docusage_user}"
POSTGRES_DB="${POSTGRES_DB:-docusage}"

MAX_RETRIES=30
RETRY_INTERVAL=2

log "Waiting for PostgreSQL at ${POSTGRES_HOST}..."

retries=0
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q; do
    retries=$((retries + 1))
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
        err "PostgreSQL not ready after ${MAX_RETRIES} attempts. Aborting."
        exit 1
    fi
    warn "PostgreSQL not ready (attempt ${retries}/${MAX_RETRIES}) — retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

log "PostgreSQL is ready."

# ── Run Migrations ────────────────────────────────────────────────
if [ "${SKIP_MIGRATIONS:-false}" != "true" ]; then
    log "Running Alembic migrations..."
    python -m alembic upgrade head
    log "Migrations complete."
else
    log "Skipping migrations (SKIP_MIGRATIONS=true)."
fi

# ── Exec main process ─────────────────────────────────────────────
log "Starting: $*"
exec "$@"
