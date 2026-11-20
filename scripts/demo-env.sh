#!/bin/sh
# Write docker/.env for docker/docker-compose.demo.yml.
#
# Starts from docker/.env.demo.example, fills every empty secret with a fresh
# random value, and takes DEMO_DOMAIN, GROQ_API_KEY and ACME_EMAIL from the
# environment when they are set. Never prints secret values.
#
# Usage, from the repo root:
#   DEMO_DOMAIN=demo.example.com GROQ_API_KEY=gsk_... ACME_EMAIL=you@example.com scripts/demo-env.sh
#   scripts/demo-env.sh --force    # overwrite an existing docker/.env
set -eu

cd "$(dirname "$0")/.."
template=docker/.env.demo.example
out=docker/.env

if [ -e "$out" ] && [ "${1:-}" != "--force" ]; then
  echo "$out already exists. Re-run with --force to replace it (this rotates every secret)." >&2
  exit 1
fi

# 48 hex characters: safe inside URLs and JSON without escaping.
random_secret() {
  od -An -N24 -tx1 /dev/urandom | tr -d ' \n'
}

tmp="$out.tmp.$$"
umask 077
trap 'rm -f "$tmp"' EXIT

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    DEMO_DOMAIN=*)
      [ -n "${DEMO_DOMAIN:-}" ] && line="DEMO_DOMAIN=$DEMO_DOMAIN" ;;
    GROQ_API_KEY=*)
      [ -n "${GROQ_API_KEY:-}" ] && line="GROQ_API_KEY=$GROQ_API_KEY" ;;
    ACME_EMAIL=*)
      [ -n "${ACME_EMAIL:-}" ] && line="ACME_EMAIL=$ACME_EMAIL" ;;
    SECRET_KEY= | POSTGRES_PASSWORD= | REDIS_PASSWORD= | QDRANT_API_KEY= | S3_ACCESS_KEY= | S3_SECRET_KEY=)
      line="$line$(random_secret)" ;;
  esac
  printf '%s\n' "$line"
done < "$template" > "$tmp"

mv "$tmp" "$out"
trap - EXIT

domain=$(sed -n 's/^DEMO_DOMAIN=//p' "$out")
echo "Wrote $out with new secrets (domain: $domain)."
if grep -q '^GROQ_API_KEY=$' "$out"; then
  echo "GROQ_API_KEY is empty. Add your key to $out before starting the stack." >&2
fi
