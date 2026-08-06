#!/usr/bin/env bash
# Install yugioh_web's JS deps. Soft-skips (exit 0) without node/pnpm.
# --frozen-lockfile because pnpm-lock.yaml is committed.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh

if ! node_env_load; then
    echo "skipping JS deps — $WEB_SKIP_REASON" >&2
    exit 0
fi

cd yugioh_web && pnpm install --frozen-lockfile
