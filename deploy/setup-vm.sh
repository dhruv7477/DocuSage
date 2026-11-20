#!/usr/bin/env bash
# One-time setup for the demo VM: Oracle Cloud Ampere A1, Ubuntu 22.04 or
# 24.04 (aarch64). Installs Docker, adds a swap file, opens ports 80 and 443
# in the host firewall, and installs the nightly database backup.
# Safe to re-run: every step checks before it changes anything.
#
# Usage, from the cloned repo:  sudo deploy/setup-vm.sh
# Options (environment): SWAP_SIZE=4G  SWAP_FILE=/swapfile  BACKUP_DIR=/var/backups/docusage
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SWAP_SIZE="${SWAP_SIZE:-4G}"
SWAP_FILE="${SWAP_FILE:-/swapfile}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/docusage}"
IPTABLES_RULES=/etc/iptables/rules.v4

log() { printf '\n== %s\n' "$*"; }

install_docker() {
  log "Docker"
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Docker and the compose plugin are already installed."
    return
  fi
  apt-get update
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  # HTTPS only, also after redirects.
  curl -fsSL --proto '=https' --tlsv1.2 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck source=/dev/null
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
  if [[ -n "${SUDO_USER:-}" ]]; then
    usermod -aG docker "$SUDO_USER"
    echo "Added $SUDO_USER to the docker group. Log out and back in to use docker without sudo."
  fi
}

add_swap() {
  log "Swap"
  if swapon --show=NAME --noheadings | grep -qxF "$SWAP_FILE"; then
    echo "$SWAP_FILE is already active."
  else
    fallocate -l "$SWAP_SIZE" "$SWAP_FILE"
    chmod 600 "$SWAP_FILE"
    mkswap "$SWAP_FILE" >/dev/null
    swapon "$SWAP_FILE"
    echo "Created a $SWAP_SIZE swap file at $SWAP_FILE."
  fi
  grep -qF "$SWAP_FILE none swap" /etc/fstab || echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
  # Swap is a safety net for build spikes, not working memory.
  echo 'vm.swappiness=10' > /etc/sysctl.d/99-docusage-swap.conf
  sysctl -q -p /etc/sysctl.d/99-docusage-swap.conf
}

# Oracle's Ubuntu images ship iptables rules that reject all inbound traffic
# except SSH. Insert ACCEPT rules for HTTP and HTTPS (and HTTP/3 over UDP)
# before the first REJECT rule, both live and in the saved rules file.
# The saved file is edited directly instead of `netfilter-persistent save`,
# which would also persist Docker's own chains.
open_port() {
  local proto="$1" port="$2"
  local spec=(-p "$proto" -m state --state NEW -m "$proto" --dport "$port" -j ACCEPT)
  if iptables -C INPUT "${spec[@]}" 2>/dev/null; then
    echo "INPUT already accepts $proto/$port."
  else
    local reject_line
    reject_line="$(iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')"
    if [[ -n "$reject_line" ]]; then
      iptables -I INPUT "$reject_line" "${spec[@]}"
    else
      iptables -A INPUT "${spec[@]}"
    fi
    echo "Opened $proto/$port."
  fi
  if [[ -f "$IPTABLES_RULES" ]]; then
    local saved="-A INPUT -p $proto -m state --state NEW -m $proto --dport $port -j ACCEPT"
    if ! grep -qxF -- "$saved" "$IPTABLES_RULES"; then
      awk -v add="$saved" '!done && /^-A INPUT -j REJECT/ {print add; done = 1} {print}' \
        "$IPTABLES_RULES" > "$IPTABLES_RULES.tmp"
      mv "$IPTABLES_RULES.tmp" "$IPTABLES_RULES"
    fi
  fi
}

open_firewall() {
  log "Host firewall"
  open_port tcp 80
  open_port tcp 443
  open_port udp 443
}

install_backup() {
  log "Nightly database backup"
  mkdir -p "$BACKUP_DIR"
  chmod 700 "$BACKUP_DIR"
  cat > /etc/cron.d/docusage-backup <<CRON
# Nightly Postgres dump of the DocuSage demo (see deploy/README.md).
SHELL=/bin/bash
30 2 * * * root BACKUP_DIR=$BACKUP_DIR $REPO_DIR/deploy/backup-db.sh >> /var/log/docusage-backup.log 2>&1
CRON
  chmod 644 /etc/cron.d/docusage-backup
  echo "Installed /etc/cron.d/docusage-backup (02:30 every night, dumps to $BACKUP_DIR)."
}

install_docker
add_swap
open_firewall
install_backup

log "Done"
echo "Next: configure and start the stack, see deploy/README.md."
