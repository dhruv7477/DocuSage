#!/bin/sh
# Start the demo stack the way CI tests it: docker/docker-compose.demo.yml
# behind Caddy on https://localhost, with the mock LLM from
# tests/e2e/compose.e2e.yml instead of Groq. Extra compose files given as
# arguments go on top (the DAST job adds tests/e2e/compose.dast.yml).
#
#   tests/e2e/start-stack.sh [extra-compose-file ...]
set -eu

DEMO_DOMAIN=localhost GROQ_API_KEY=ci-not-a-real-key scripts/demo-env.sh
# The mock LLM is another container on the compose network, not the internet.
sed -i 's|^LLM_BASE_URL=.*|LLM_BASE_URL=http://mock-llm:8080/v1|' docker/.env

COMPOSE_FILE=docker/docker-compose.demo.yml:tests/e2e/compose.e2e.yml
for extra in "$@"; do COMPOSE_FILE="$COMPOSE_FILE:$extra"; done
export COMPOSE_FILE COMPOSE_PATH_SEPARATOR=:  # ":" on every OS
docker compose up -d --build --wait --wait-timeout 900
