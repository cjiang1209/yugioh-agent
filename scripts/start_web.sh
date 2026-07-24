#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_ROOT/yugioh_web"

if [ ! -d "$WEB_DIR/dist" ]; then
    echo "Error: dist/ not found. Run scripts/build_web.sh first."
    exit 1
fi

# ─── Ensure nvm + node are available ─────────────────────────────────────────
if ! command -v node &>/dev/null; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        source "$HOME/.nvm/nvm.sh"
    else
        echo "Error: node not found and nvm is not installed"
        exit 1
    fi
fi

cd "$WEB_DIR"

mode="${1:-production}"

case "$mode" in
    --dev)
        echo "Starting web UI in development mode ..."
        exec pnpm dev
        ;;
    *)
        echo "Starting web UI (production) on port ${PORT:-7000} ..."
        export PORT="${PORT:-7000}"
        exec pnpm start
        ;;
esac
