# Run the demo on an Oracle Cloud Always Free VM

This guide puts `docker/docker-compose.demo.yml` on one Oracle Cloud Ampere A1 VM (linux/arm64), behind Caddy with automatic HTTPS. The LLM is hosted (Groq by default), and everything else runs on the VM.

Nothing here creates cloud resources for you. You click through the console steps yourself, then run the scripts on the VM.

## What you need

- An Oracle Cloud account with Always Free Ampere capacity in your home region.
- A domain name you control, for `DEMO_DOMAIN`. The examples use `demo.example.com`.
- A Groq API key from https://console.groq.com/keys.

## 1. Create the VM (Oracle Cloud console)

1. Compute > Instances > Create instance.
2. Image: Canonical Ubuntu 24.04 (aarch64). Shape: `VM.Standard.A1.Flex` with 2 OCPU and 12 GB memory. Both fit in the Always Free allowance.
3. Boot volume: the default 47 GB is enough. The images and models use about 4 GB.
4. Add your SSH public key and create the instance. If the console reports "Out of capacity", retry later or pick another availability domain. Free Ampere capacity is often exhausted.
5. Note the public IP address.

## 2. Open the network (Oracle Cloud console)

The VCN security list is the outer firewall. In Networking > Virtual Cloud Networks > your VCN > Security Lists > Default Security List, add these ingress rules:

| Source | Protocol | Port | Why |
|---|---|---|---|
| your own IP, as `x.x.x.x/32` | TCP | 22 | SSH (tighten the default `0.0.0.0/0` rule) |
| `0.0.0.0/0` | TCP | 80 | Let's Encrypt challenge, redirect to HTTPS |
| `0.0.0.0/0` | TCP | 443 | the demo |
| `0.0.0.0/0` | UDP | 443 | HTTP/3 (optional) |

Open nothing else. Postgres, Redis, Qdrant, SeaweedFS and the API are not published on the host at all.

## 3. Point DNS at the VM

At your DNS provider, create an `A` record for `demo.example.com` with the VM's public IP. Wait until `dig +short demo.example.com` returns that IP. Caddy cannot get a certificate before it does.

## 4. Prepare the VM

```bash
ssh ubuntu@<public-ip>
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/dhruv7477/DocuSage.git
cd DocuSage
sudo deploy/oracle/setup-vm.sh
```

`setup-vm.sh` installs Docker, adds a 4 GB swap file, opens ports 80 and 443 in the host firewall (Oracle's Ubuntu images reject everything except SSH by default), and installs a nightly database backup. It is safe to run again. Log out and back in afterwards, so your user can run `docker` without `sudo`.

## 5. Configure and start

```bash
DEMO_DOMAIN=demo.example.com GROQ_API_KEY=gsk_your_key scripts/demo-env.sh
docker compose -f docker/docker-compose.demo.yml up -d --build
```

`demo-env.sh` writes `docker/.env` with random secrets for Postgres, Redis, Qdrant, the S3 store and JWT signing. To rotate them all later, run it with `--force`, then `docker compose ... down -v` (this deletes the demo data) and `up -d` again.

The first start takes about 10 minutes on 2 OCPUs: two image builds plus about 220 MB of model downloads. Follow it with:

```bash
docker compose -f docker/docker-compose.demo.yml ps
docker compose -f docker/docker-compose.demo.yml logs -f api caddy
```

## 6. Check it

```bash
curl -s https://demo.example.com/health
# {"status":"ok","app":"DocuSage","env":"production"}
```

Then open `https://demo.example.com` and sign up. The sign-up page shows the demo limits, and the new account gets a "Sample documents" workspace with two public-domain PDFs (they are ingested once when the worker first starts, which takes a few minutes). Ask a question about them, or upload your own PDF.

## 7. Uptime check

Use any free external monitor, for example UptimeRobot, Better Stack or Healthchecks.io. Create an HTTPS keyword monitor:

- URL: `https://demo.example.com/health`
- Interval: 5 minutes
- Expect status 200 and the keyword `"status":"ok"`
- Alert: your email

`/health` answers without touching the database or the LLM, so it does not spend Groq quota. `/health/deep` is not reachable through Caddy.

## 8. Backups and restore

The cron job from `setup-vm.sh` runs `deploy/oracle/backup-db.sh` at 02:30 UTC. It writes `/var/backups/docusage/docusage-<timestamp>.dump` and keeps 7 days of dumps. Its log is `/var/log/docusage-backup.log`.

The dump covers accounts, workspaces, chats and document records. Uploaded files and vectors are not backed up: the demo deletes every upload each night at 03:00 UTC (`DEMO_WIPE_HOUR_UTC`), and the sample documents are rebuilt from `demo/samples/` if they go missing. Copy dumps off the VM if you want to keep them.

To restore a dump:

```bash
docker compose -f docker/docker-compose.demo.yml exec -T postgres \
  pg_restore -U docusage -d docusage --clean --if-exists < /var/backups/docusage/docusage-<timestamp>.dump
```

## 9. Update to a new version

```bash
cd DocuSage
git pull
docker compose -f docker/docker-compose.demo.yml up -d --build
```

Database migrations run automatically when the API container starts.

## Troubleshooting

- **Browser shows a certificate error.** DNS did not point at the VM when Caddy first started, or port 80 is closed. Fix it, then run `docker compose -f docker/docker-compose.demo.yml restart caddy`.
- **The site times out from outside, but `curl -k https://localhost` works on the VM.** Check both firewalls: the VCN security list (step 2) and `sudo iptables -L INPUT -n`.
- **Answers fail with an LLM error.** Groq's free tier limits requests and tokens per minute per model. Check your limits in the Groq console, or set smaller models in `docker/.env` and run `docker compose ... up -d` again.
