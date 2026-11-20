# ── Build Stage ────────────────────────────────────────────────────
FROM python:3.15-slim AS builder

WORKDIR /src

COPY requirements.txt constraints.txt ./
# torch comes from the CPU-only index first. On amd64, PyPI's torch wheel pulls
# several GB of CUDA libraries this app never uses. The version is read from
# constraints.txt so it is pinned in one place. Wheels only, so no package's
# setup script runs during the build.
RUN pip install --no-cache-dir --only-binary :all: "$(grep '^torch==' constraints.txt)" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --only-binary :all: -r requirements.txt -c constraints.txt \
    && pip uninstall -y pip

# ── Runtime Stage ──────────────────────────────────────────────────
FROM python:3.15-slim AS runtime

WORKDIR /src

# upgrade: security fixes released since the base image was built.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y pip

# Copy installed packages from builder. Neither stage keeps pip: it is not
# needed at run time, and it vendors old copies of other packages that
# scanners report.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only what the API and the worker run: the code, the Alembic config
# (it points at src/db/migrations) and the demo's sample documents. Tests,
# CI scripts and deploy files stay out of the image.
COPY src ./src
COPY alembic.ini ./
COPY demo ./demo

# Copy and set entrypoint
COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root user for security. The Hugging Face cache directory exists in the
# image so a named volume mounted there inherits appuser ownership.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN useradd -m -u 1000 appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /src /home/appuser/.cache
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
# One worker: each one loads its own embedder and reranker (about 450 MB) and
# keeps its own in-process state. Behind a proxy, set FORWARDED_ALLOW_IPS so
# uvicorn trusts the proxy's X-Forwarded-For header.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
