#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_ROOT/yugioh_web"

if [ ! -d "$WEB_DIR" ]; then
    echo "Error: yugioh_web/ not found at $WEB_DIR"
    exit 1
fi

# ─── Ensure nvm + node are available ─────────────────────────────────────────
if ! command -v node &>/dev/null; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        source "$HOME/.nvm/nvm.sh"
    else
        echo "Error: node not found and nvm is not installed"
        echo "Install nvm: https://github.com/nvm-sh/nvm"
        exit 1
    fi
fi

if ! command -v pnpm &>/dev/null; then
    echo "Error: pnpm not found"
    echo "Install: npm install -g pnpm"
    exit 1
fi

echo "Using node $(node --version), pnpm $(pnpm --version)"

# ─── Install dependencies ────────────────────────────────────────────────────
echo "Installing dependencies ..."
cd "$WEB_DIR"
pnpm install

# ─── Build ───────────────────────────────────────────────────────────────────
echo "Building web UI ..."
pnpm build

echo "Done! Run scripts/start_web.sh to start the server."
