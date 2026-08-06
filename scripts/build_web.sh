#!/usr/bin/env bash
# Compile the web bundle. Assumes JS deps are installed (make install-web).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh

node_env_load || { echo "error: $WEB_SKIP_REASON" >&2; exit 1; }
cd yugioh_web && pnpm build
echo "Done! Run scripts/start_web.sh to start the server."
