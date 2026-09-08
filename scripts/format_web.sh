#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh
web_guard "web format (prettier)"
cd web && exec pnpm format
