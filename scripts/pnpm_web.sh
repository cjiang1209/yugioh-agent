#!/usr/bin/env bash
# Run pnpm in yugioh_web/ with node/pnpm on PATH. pre-commit runs hook entries
# without a login shell, so nvm isn't loaded and a bare `pnpm` isn't found;
# node_env.sh sources it.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/node_env.sh
node_env_load || { echo "error: $WEB_SKIP_REASON" >&2; exit 1; }
cd yugioh_web && exec pnpm "$@"
