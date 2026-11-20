#!/usr/bin/env bash
# Dump the demo's Postgres database and keep the last KEEP_DAYS days of dumps.
# Installed as a nightly cron job by setup-vm.sh. Can also be run by hand.
#
# Usage: deploy/backup-db.sh
# Options (environment): BACKUP_DIR=/var/backups/docusage  KEEP_DAYS=7
# Restore:  docker compose -f docker/docker-compose.demo.yml exec -T postgres \
#             pg_restore -U docusage -d docusage --clean --if-exists < FILE.dump
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/docusage}"
KEEP_DAYS="${KEEP_DAYS:-7}"

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
target="$BACKUP_DIR/docusage-$stamp.dump"

# Write to a temporary name first so a failed dump never looks like a backup.
umask 077
docker compose -f "$REPO_DIR/docker/docker-compose.demo.yml" exec -T postgres \
  pg_dump -U docusage -d docusage --format=custom > "$target.partial"
mv "$target.partial" "$target"

find "$BACKUP_DIR" -name 'docusage-*.dump' -type f -mtime +"$KEEP_DAYS" -delete
echo "$(date -u +%FT%TZ) wrote $target ($(du -h "$target" | cut -f1))"
