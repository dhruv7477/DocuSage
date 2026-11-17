#!/bin/sh
# Start SeaweedFS (master, volume, filer and S3 gateway in one process) with a
# single S3 identity taken from S3_ACCESS_KEY and S3_SECRET_KEY.
# Used by docker/docker-compose.yml and docker/docker-compose.demo.yml.
set -eu

: "${S3_ACCESS_KEY:?S3_ACCESS_KEY is required}"
: "${S3_SECRET_KEY:?S3_SECRET_KEY is required}"

umask 077
cat > /tmp/s3.json <<JSON
{"identities": [{"name": "app",
  "credentials": [{"accessKey": "${S3_ACCESS_KEY}", "secretKey": "${S3_SECRET_KEY}"}],
  "actions": ["Admin", "Read", "Write", "List", "Tagging"]}]}
JSON

exec weed server \
  -dir=/data \
  -ip.bind=0.0.0.0 \
  -s3 -s3.port=8333 -s3.config=/tmp/s3.json \
  -master.volumeSizeLimitMB=1024 \
  -volume.max=0
