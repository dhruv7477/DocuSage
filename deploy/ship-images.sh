#!/usr/bin/env bash
# Build the demo images on this machine and load them on the server.
#
# The server (an AWS t4g.small: 2 vCPU, 2 GB) can build them itself, but
# slowly and only with the stack stopped (see deploy/README.md). This
# machine's Docker engine must be linux/arm64 like the server: Apple
# silicon, or an arm64 Windows or Linux machine.
#
# Usage, from the repo root:  deploy/ship-images.sh SSH_TARGET
#   SSH_TARGET  what you pass to ssh: user@host, or a Host from ~/.ssh/config
# Options (environment):
#   SSH_OPTS     extra ssh options, for example "-i ~/.ssh/docusage.pem"
#   REMOTE_REPO  the repo on the server (default: DocuSage, in the login's home)
#
# Safe to re-run: it rebuilds from the build cache and loads again.
set -euo pipefail

TARGET="${1:?usage: deploy/ship-images.sh SSH_TARGET (user@host or an ssh Host alias)}"
REMOTE_REPO="${REMOTE_REPO:-DocuSage}"
ARCHIVE_NAME=docusage-images.tar.gz
IMAGES=(docusage-api:demo docusage-web:demo)
# Everything the two Dockerfiles copy into the images.
IMAGE_INPUTS=(src demo docker requirements.txt constraints.txt alembic.ini
  package.json package-lock.json tsconfig.json tsconfig.node.json
  vite.config.ts tailwind.config.js postcss.config.js)

cd "$(dirname "$0")/.."
read -r -a ssh_opts <<< "${SSH_OPTS:-}"
# The remote commands are built here, with local values, on purpose.
# shellcheck disable=SC2029
remote() { ssh "${ssh_opts[@]}" "$TARGET" "$@"; }
log() { printf '\n== %s\n' "$*"; }

log "Checks"
arch="$(docker info --format '{{.Architecture}}')"
if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
  echo "This Docker engine is $arch. Build on an arm64 machine, or on the server (deploy/README.md)." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain -- "${IMAGE_INPUTS[@]}")" ]]; then
  echo "Uncommitted changes in files that go into the images. Commit them (and pull on the server) first:" >&2
  git status --short -- "${IMAGE_INPUTS[@]}" >&2
  exit 1
fi
commit="$(git rev-parse HEAD)"
remote_commit="$(remote "git -C '$REMOTE_REPO' rev-parse HEAD")"
if [[ "$remote_commit" != "$commit" ]]; then
  echo "The server's checkout is at ${remote_commit:0:12}, this one at ${commit:0:12}." >&2
  echo "Check out the same commit in both places, so the compose file matches the images." >&2
  exit 1
fi
echo "Engine $arch, commit ${commit:0:12} here and on the server."

log "Build"
build() {
  docker build --platform linux/arm64 -f "$1" -t "$2" \
    --label "org.opencontainers.image.revision=$commit" .
}
build docker/Dockerfile docusage-api:demo
build docker/Dockerfile.web docusage-web:demo

log "Save and compress"
archive="$(mktemp -d)/$ARCHIVE_NAME"
trap 'rm -rf "$(dirname "$archive")"' EXIT
docker save "${IMAGES[@]}" | gzip -6 > "$archive"
sum="$(sha256sum "$archive" | cut -d' ' -f1)"
echo "$(du -h "$archive" | cut -f1) ($sum)"

log "Upload"
remote "cat > '$ARCHIVE_NAME.partial'" < "$archive"
remote_sum="$(remote "sha256sum '$ARCHIVE_NAME.partial'" | cut -d' ' -f1)"
if [[ "$remote_sum" != "$sum" ]]; then
  echo "The upload is corrupt (checksum $remote_sum). Run the script again." >&2
  exit 1
fi

log "Load on the server"
# Dangling images are the previous versions these tags pointed to.
remote "gunzip -c '$ARCHIVE_NAME.partial' | docker load && rm -f '$ARCHIVE_NAME.partial' && docker image prune -f >/dev/null"
for image in "${IMAGES[@]}"; do
  loaded="$(remote "docker image inspect --format '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' $image")"
  if [[ "$loaded" != "$commit" ]]; then
    echo "$image on the server is from ${loaded:0:12}, expected ${commit:0:12}." >&2
    exit 1
  fi
done

log "Done"
echo "Both images are on the server at ${commit:0:12}. Start or update the stack there with:"
echo "  docker compose -f docker/docker-compose.demo.yml up -d --no-build"
