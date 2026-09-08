#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh
web_guard "web lint (eslint)"
cd web && exec pnpm lint
