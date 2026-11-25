# Run the demo on one small arm64 server

This guide puts `docker/docker-compose.demo.yml` on one linux/arm64 server behind Caddy with automatic HTTPS. The LLM is hosted (Groq by default); everything else runs on the server.

The main path is an **AWS EC2 t4g.small** (2 vCPU Graviton, 2 GB of RAM, Ubuntu 24.04). An **Oracle Cloud** Ampere VM works too; its steps are in [Alternative: Oracle Cloud](#alternative-oracle-cloud). After the host steps, both follow the same steps from [Prepare the server](#prepare-the-server).

Nothing here creates cloud resources for you. You click through the console steps yourself, then run the scripts on the server.

The examples use placeholders. Replace them with your own values, and keep the real ones out of the repository:

| Placeholder | Means |
|---|---|
| `demo.example.com` | the demo's host name (a subdomain you control) |
| `x.x.x.x` | your computer's public IPv4 address |
| `<elastic-ip>` | the server's Elastic IP address |
| `you@example.com` | your email address |

## What you need

- An AWS account (the Free plan is enough for a t4g.small), or an Oracle Cloud account.
- A domain whose DNS you manage. The steps use Cloudflare; any DNS provider works.
- A Groq API key from https://console.groq.com/keys.
- Optional but recommended: a computer whose Docker engine is linux/arm64 (Apple silicon, or an arm64 Windows or Linux machine), to build the images there instead of on the 2 GB server. `docker info --format '{{.Architecture}}'` prints `aarch64` on one.

## AWS EC2 (t4g.small)

### 1. Key pair

EC2 > Network & Security > Key pairs > Create key pair: type ED25519, format `.pem`. Your browser downloads it once; keep it, for example as `docusage.pem` in your `.ssh` folder.

SSH refuses a private key that other users can read. On macOS or Linux:

```bash
chmod 400 ~/.ssh/docusage.pem
```

On Windows, in PowerShell or Command Prompt, remove the inherited permissions and give only your own user read access (without this, `ssh` stops with "UNPROTECTED PRIVATE KEY FILE" or "Permissions ... are too open"):

```powershell
icacls $env:USERPROFILE\.ssh\docusage.pem /inheritance:r
icacls $env:USERPROFILE\.ssh\docusage.pem /grant:r "$($env:USERNAME):(R)"
icacls $env:USERPROFILE\.ssh\docusage.pem
```

The last command should list only your user, with `(R)`.

### 2. Security group

EC2 > Network & Security > Security groups > Create security group, in the VPC you will launch into. Inbound rules:

| Type | Protocol | Port | Source | Why |
|---|---|---|---|---|
| SSH | TCP | 22 | `x.x.x.x/32` | SSH from your computer only |
| HTTP | TCP | 80 | `0.0.0.0/0` | Let's Encrypt challenge, redirect to HTTPS |
| HTTPS | TCP | 443 | `0.0.0.0/0` | the demo |
| Custom UDP | UDP | 443 | `0.0.0.0/0` | HTTP/3 (optional) |

Keep the default outbound rule (all traffic). Open nothing else: Postgres, Redis, Qdrant, SeaweedFS and the API are not published on the host at all.

For the SSH rule, look up your public **IPv4** address and enter it with `/32`:

```bash
curl -4 https://checkip.amazonaws.com
```

> **Do not rely on the console's "My IP" option.** When your browser reaches the AWS console over IPv6, "My IP" fills in your IPv6 address (a `/128`). The instance has only an IPv4 address, so your SSH connection arrives over IPv4, never matches that rule, and simply times out (no "connection refused"). Check that the SSH rule's source is an IPv4 `/32`. Your home IPv4 address can also change; if SSH starts timing out later, update the rule.

### 3. Launch the instance

EC2 > Instances > Launch instances:

- Name: `docusage-demo`.
- AMI: **Ubuntu Server 24.04 LTS**, architecture **64-bit (Arm)**.
- Instance type: **t4g.small** (2 vCPU, 2 GiB).
- Key pair: the one from step 1.
- Network settings > Select existing security group: the one from step 2.
- Storage: **40 GiB gp3**. The images, models and build cache use about 8 GB; the 4 GB swap file is created on it later.

T4g instances run in Unlimited CPU credit mode by default, so a long burst above the baseline (such as building the images on the server) can cost extra. Building on your own computer (see [Images](#images)) avoids that. To rule it out, set Actions > Instance settings > Change credit specification > Standard.

### 4. Elastic IP

The automatic public IP changes every time the instance stops and starts. Give it a fixed address instead: EC2 > Network & Security > Elastic IPs > Allocate Elastic IP address, then Actions > Associate Elastic IP address > your instance. This is `<elastic-ip>` below.

AWS bills public IPv4 addresses, Elastic IPs included, so check the Billing console. An Elastic IP that is not associated with a running instance is billed too: release it when you tear the demo down.

### 5. DNS record (Cloudflare)

In Cloudflare, your domain > DNS > Records > Add record:

| Type | Name | IPv4 address | Proxy status | TTL |
|---|---|---|---|---|
| A | `demo` (for `demo.example.com`) | `<elastic-ip>` | **DNS only** (grey cloud) | Auto |

Use DNS only, not Proxied: Caddy on the server gets its own Let's Encrypt certificate on port 80 and terminates HTTPS itself. Wait until the name resolves to the Elastic IP:

```bash
nslookup demo.example.com 1.1.1.1
```

Caddy cannot get a certificate before it does.

### 6. Connect

The login user on Ubuntu AMIs is `ubuntu`:

```bash
ssh -i ~/.ssh/docusage.pem ubuntu@<elastic-ip>
```

An entry in `~/.ssh/config` (on Windows, `%USERPROFILE%\.ssh\config`) saves typing, and `deploy/ship-images.sh` can use it:

```
Host docusage-demo
    HostName <elastic-ip>
    User ubuntu
    IdentityFile ~/.ssh/docusage.pem
```

Then `ssh docusage-demo`. If SSH times out, check the security group's SSH source (step 2) first.

Continue with [Prepare the server](#prepare-the-server).

## Alternative: Oracle Cloud

An Oracle Cloud Ampere A1 VM (`VM.Standard.A1.Flex`) works the same way, and the Always Free allowance gives it more memory than the t4g.small.

1. **Create the VM.** Compute > Instances > Create instance. Image: Canonical Ubuntu 24.04 (aarch64). Shape: `VM.Standard.A1.Flex` with 2 OCPU and 12 GB memory. Add your SSH public key. If the console reports "Out of capacity", retry later or pick another availability domain. Note the public IP address; the login user is `ubuntu`.
2. **Open the network.** The VCN security list is the outer firewall. In Networking > Virtual Cloud Networks > your VCN > Security Lists > Default Security List, add ingress rules for TCP 80, TCP 443 and UDP 443 from `0.0.0.0/0`, and narrow the SSH rule to your IPv4 as `x.x.x.x/32`. Oracle's Ubuntu images also have host firewall rules that reject everything except SSH; `setup-vm.sh` opens 80 and 443 in them.
3. **Point DNS at the VM** with an `A` record, as in [step 5](#5-dns-record-cloudflare) above.

## Prepare the server

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/dhruv7477/DocuSage.git
cd DocuSage
sudo deploy/setup-vm.sh
```

`setup-vm.sh` installs Docker and the compose plugin, adds a 4 GB swap file (`/swapfile`, swappiness 10), opens ports 80 and 443 in the host firewall where the image rejects them (Oracle; on AWS the security group is the firewall and nothing is changed), and installs the nightly database backup. It checks before every change, so running it again changes nothing. Log out and back in afterwards, so your user can run `docker` without `sudo`.

## Images

The demo runs two images, `docusage-api:demo` (about 2.5 GB) and `docusage-web:demo` (about 170 MB). Pick one of these.

### Recommended: build on your arm64 computer and copy them over

The 2 GB server never has to build anything, and the image it runs is exactly the one you built. On your computer, with the repository checked out at the **same commit as the server**:

```bash
deploy/ship-images.sh docusage-demo
# or, without the ~/.ssh/config entry:
SSH_OPTS="-i ~/.ssh/docusage.pem" deploy/ship-images.sh ubuntu@<elastic-ip>
```

On Windows, run it in Git Bash (it comes with Git for Windows and has `bash`, `ssh`, `gzip` and `sha256sum`).

The script checks that your Docker engine is arm64, that nothing that goes into the images is uncommitted, and that the server's checkout (`~/DocuSage`, or `REMOTE_REPO`) is at the same commit. It builds both images with the commit as a label, compresses them (about 560 MB), uploads the archive over SSH, checks its SHA-256 on the server, runs `docker load` there, and checks the label on the loaded images. A failed or interrupted run leaves the server's images as they were; run it again. With a warm build cache it takes about 2 minutes plus the upload: 560 MB is about 4 minutes at 20 Mbit/s.

Then, on the server, use `--no-build` so Compose uses the loaded images:

```bash
docker compose -f docker/docker-compose.demo.yml up -d --no-build
```

### Or: build on the server

This works too, but only with the stack stopped: the build and a running stack together do not fit in 2 GB. A cold build of both images, on a stand-in with 2 CPUs, 2 GB of RAM and 4 GB of swap, took about 12 minutes (5.5 for the API image, 7 for the web image, which compiles Caddy) and used all 2 GB plus a little swap. The stand-in had faster cores than a t4g.small, so expect longer, and the build uses CPU credits (see step 3). On the server:

```bash
docker compose -f docker/docker-compose.demo.yml down     # keeps the data volumes
docker compose -f docker/docker-compose.demo.yml build
docker compose -f docker/docker-compose.demo.yml up -d
```

## First boot checklist

**1. Write `docker/.env`** (once):

```bash
DEMO_DOMAIN=demo.example.com GROQ_API_KEY=gsk_your_key ACME_EMAIL=you@example.com scripts/demo-env.sh
```

| Setting | Value |
|---|---|
| `DEMO_DOMAIN` | the host name from the DNS step, without `https://` |
| `GROQ_API_KEY` | your Groq key |
| `ACME_EMAIL` | where the certificate authority can reach you about the certificate. Optional, but with it Caddy can also fall back to ZeroSSL if Let's Encrypt fails. |
| `SECRET_KEY`, `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `QDRANT_API_KEY`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` | generated by the script; nothing to do |
| `LLM_MODEL`, `ROUTER_MODEL`, `SUMMARY_MODEL` | current Groq model IDs; the defaults are `openai/gpt-oss-120b` and `openai/gpt-oss-20b` |

The file is only readable by you. To rotate every secret later, run the script with `--force`, then `docker compose -f docker/docker-compose.demo.yml down -v` (this deletes the demo data) and start again.

**2. Start the stack**, with `--no-build` if you copied the images over, or with `--build` to build on the server:

```bash
docker compose -f docker/docker-compose.demo.yml up -d --no-build
docker compose -f docker/docker-compose.demo.yml ps
```

The first start downloads the embedding and reranker models (about 220 MB); the API turns healthy after a few minutes. Every service should show `healthy` or `running`.

**3. Check the HTTPS certificate:**

```bash
curl -sI https://demo.example.com/health | head -1
echo | openssl s_client -connect demo.example.com:443 -servername demo.example.com 2>/dev/null \
  | openssl x509 -noout -issuer -dates
```

The first command prints `HTTP/2 200`; the second shows Let's Encrypt (or ZeroSSL) as the issuer and the validity dates. If either fails, see why Caddy could not get the certificate:

```bash
docker compose -f docker/docker-compose.demo.yml logs caddy | grep -i -E 'certificate|acme|error'
```

The usual causes are DNS not yet pointing at the server, port 80 closed in the security group, or Cloudflare set to Proxied. Fix it, then `docker compose -f docker/docker-compose.demo.yml restart caddy`.

**4. Check memory:**

```bash
docker stats --no-stream
free -m
docker inspect -f '{{.Name}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}' $(docker ps -q)
```

Each container stays under its limit (the `MEM USAGE / LIMIT` column; see [Memory](#memory-on-a-2-gb-server)). Some swap in use is expected: the API, the worker and Qdrant page out memory they do not use. Every container should show `oom=false restarts=0`.

**5. Check the first real Groq answer.** Open `https://demo.example.com`, sign up, open the "Sample documents" workspace (its two documents are ready after a few minutes on first start), and ask:

> When must an employer complete Section 2 of Form I-9?

A cited answer streams in. If it does not, see what the API reported:

```bash
docker compose -f docker/docker-compose.demo.yml logs api | grep -i -E 'llm|groq|429|error' | tail -20
```

**6. If gpt-oss answers come back empty:** gpt-oss writes its reasoning first, and the reasoning counts toward the reply's token budget. When it uses the whole budget, the answer is empty. Keep `LLM_REASONING_EFFORT=low` (the default), and if answers are still empty, raise `LLM_MAX_TOKENS` (default 2048) in `docker/.env`, for example to 4096, then recreate the API and the worker:

```bash
echo "LLM_MAX_TOKENS=4096" >> docker/.env
docker compose -f docker/docker-compose.demo.yml up -d --no-build
```

A larger budget uses more of Groq's free tier per answer (see [Limits](#limits-and-the-groq-free-tier)).

## Memory on a 2 GB server

Every service has a memory limit, and the limits add up to 1560 MB of the 2 GB. That leaves room for the kernel, Docker and SSH. The CI job "Demo stack end to end (arm64)" runs the stack with these limits, fails if any process is OOM-killed, and reports each container's peak in its job summary.

| Service | Memory limit | Swap allowed | What it holds |
|---|---|---|---|
| api | 600 MB | 512 MB | one uvicorn process: the embedder and the reranker |
| worker | 600 MB | 512 MB | one Celery process (replaced after 10 tasks) plus beat: parsing and the embedder |
| qdrant | 128 MB | 512 MB | the vectors, one collection per workspace |
| postgres | 96 MB | none | `shared_buffers` 24 MB, 50 connections |
| seaweedfs | 64 MB | none | the uploaded files |
| caddy | 40 MB | none | HTTPS and the frontend |
| redis | 32 MB | none | the task queue and rate limits (16 MB, never evicts) |

Only the API, the worker and Qdrant may use swap. The two Python services import about 150 MB each that they never use again, and Qdrant holds collections nobody is searching: that is what pages out. A leak in any other service ends in an OOM kill and a restart, not a slow server. The models run in batches of 8 (`EMBEDDING_BATCH_SIZE`, `RERANKER_BATCH_SIZE`), which for long texts needs 240 to 280 MB less per model than batches of 32. `MALLOC_ARENA_MAX=2` keeps the Python heaps compact.

Qdrant grows with the number of accounts, because each account gets its own copy of the sample documents: about 3 MB per account. It holds about 20 accounts in its 128 MB and roughly 190 with its swap. Beyond that it needs a larger limit, or accounts (and their sample copies) removed after a while.

## Uptime check

Use any free external monitor, for example UptimeRobot, Better Stack or Healthchecks.io. Create an HTTPS keyword monitor:

- URL: `https://demo.example.com/health`
- Interval: 5 minutes
- Expect status 200 and the keyword `"status":"ok"`
- Alert: your email

`/health` answers without touching the database or the LLM, so it does not spend Groq quota. `/health/deep` is not reachable through Caddy.

## Backups and restore

The cron job from `setup-vm.sh` runs `deploy/backup-db.sh` at 02:30 UTC. It writes `/var/backups/docusage/docusage-<timestamp>.dump` and keeps 7 days of dumps. Its log is `/var/log/docusage-backup.log`.

The dump covers accounts, workspaces, chats and document records. Uploaded files and vectors are not backed up: the demo deletes every upload each night at 03:00 UTC (`DEMO_WIPE_HOUR_UTC`), and the sample documents are rebuilt from `demo/samples/` if they go missing. Copy dumps off the server if you want to keep them.

To restore a dump:

```bash
docker compose -f docker/docker-compose.demo.yml exec -T postgres \
  pg_restore -U docusage -d docusage --clean --if-exists < /var/backups/docusage/docusage-<timestamp>.dump
```

## Update to a new version

On the server, `git pull`. Then either copy new images over from your computer (after `git pull` there too, so both are at the same commit):

```bash
deploy/ship-images.sh docusage-demo                           # on your computer
docker compose -f docker/docker-compose.demo.yml up -d --no-build   # on the server
```

or build on the server, with the stack stopped as described in [Images](#images). Database migrations run automatically when the API container starts.

### When a release changes how documents are chunked

Then the sample documents need re-ingesting once, after the new containers are up. The pull request says so when it applies (the first one is the release that puts headings into chunk text). Uploads are left alone: the nightly wipe removes them, and new uploads get the new chunks.

```bash
docker compose -f docker/docker-compose.demo.yml exec worker \
  celery -A src.workers.celery_app call src.workers.demo_tasks.refresh_sample_library_task
docker compose -f docker/docker-compose.demo.yml logs -f --since 1m worker | grep sample_library_refresh
```

The first command prints a task ID and returns. The worker then re-ingests the sample library and copies it again into every account's "Sample documents". The log line `sample_library_refreshed documents=2 copies=N` means it finished; press Ctrl+C to stop following the log. It took 34 seconds on a laptop; allow a few minutes on the t4g.small.

- Run it at a quiet time: while it runs, a question about the samples can find nothing, and new accounts wait for their sample copies.
- `sample_library_refresh_failed` means a sample failed to ingest. The accounts' copies are not touched then; look at `docker compose -f docker/docker-compose.demo.yml logs worker` and run the command again.
- Running it again is always safe.

## Limits and the Groq free tier

The demo defaults are sized for Groq's free tier for `openai/gpt-oss-120b`: about 8K tokens per minute and 200K tokens per day. Check your own limits on the Limits page of the Groq console, because they differ per model and change over time. An answer uses about 3.5K tokens (up to 2.5K of document context, up to 1K of chat history, the prompt and the reply), so the whole demo gets roughly 50 questions a day and 2 a minute.

| Setting | Demo default | What it does |
|---|---|---|
| `QUOTA_QUESTIONS_PER_DAY` | 10 | Questions per user per UTC day |
| `DEMO_GLOBAL_QUESTIONS_PER_DAY` | 50 | Questions per UTC day for all visitors together. After that, visitors see "The demo has reached today's limit. Try again tomorrow, or self-host it from the README." Admins are not blocked. |
| `RATE_LIMIT_CHAT_PER_MINUTE` | 10 | Questions per user per minute |
| `DOC_SUMMARY_ENABLED` | false | Whole-document summaries. A summary of 40 chunks is about 13K tokens in under a minute (four map calls of about 2.8K each and a final pass), more than 8K per minute on its own. When off, a summary request gets a normal cited answer and a note that summaries are off. |
| `SUMMARY_MAX_TOTAL_CHUNKS` | 40 | Chunks per summary, when summaries are on |
| `LLM_REASONING_EFFORT` | low | Reasoning effort for gpt-oss, whose reasoning tokens count toward the limits |
| `LLM_MAX_TOKENS` | 2048 | Tokens per reply, reasoning included |
| `QUOTA_MAX_DOCUMENTS`, `QUOTA_MAX_UPLOAD_MB` | 5, 25 | Uploaded documents and megabytes per user. Sample documents do not count. |

On a paid tier, set new values in `docker/.env` and run `docker compose -f docker/docker-compose.demo.yml up -d --no-build`:

- Set `DEMO_GLOBAL_QUESTIONS_PER_DAY` to about your tokens per day divided by 4,000, which leaves some headroom, or 0 to turn it off.
- Turn summaries on with `DOC_SUMMARY_ENABLED=true` once your per-minute limit is well above 13K tokens.
- Raise `QUOTA_QUESTIONS_PER_DAY` and `RATE_LIMIT_CHAT_PER_MINUTE` as you like. 0 turns a quota off.

## Troubleshooting

- **SSH times out.** The security group's SSH source is not your current public IPv4 as `x.x.x.x/32`: "My IP" may have filled in an IPv6 address, or your IPv4 address changed. See step 2.
- **Browser shows a certificate error.** DNS did not point at the server when Caddy first started, port 80 is closed, or the Cloudflare record is Proxied instead of DNS only. Fix it, then run `docker compose -f docker/docker-compose.demo.yml restart caddy`.
- **The site times out from outside, but `curl -k https://localhost` works on the server.** Check the cloud firewall (the AWS security group, or Oracle's VCN security list) and, on Oracle, `sudo iptables -L INPUT -n`.
- **A container keeps restarting.** `docker inspect -f '{{.State.OOMKilled}}' <container>` shows whether it ran out of memory, and `docker compose -f docker/docker-compose.demo.yml logs --tail=100 <service>` shows why it stopped.
- **Answers fail with an LLM error.** Groq's free tier limits requests and tokens per minute per model; see [Limits](#limits-and-the-groq-free-tier). Also check https://console.groq.com/docs/deprecations: a model that has been shut down fails every call. Set current model IDs in `docker/.env` and run `docker compose ... up -d` again.
- **The API container does not start and its log shows `llm` not ready.** The key is wrong, or `LLM_MODEL` is not in the provider's model list.
