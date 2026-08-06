#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh
web_guard "web lint (eslint)"
cd yugioh_web && exec pnpm lint
