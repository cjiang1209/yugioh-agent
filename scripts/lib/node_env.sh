#!/usr/bin/env bash
# Node/pnpm discovery + the web-suite skip contract, in one place. Source this
# file; on failure the functions set WEB_SKIP_REASON.

WEB_DIR="${WEB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/yugioh_web}"
WEB_SKIP_REASON=""

# Put node + pnpm on PATH, sourcing nvm if node isn't already there (nvm does
# not load in a non-interactive shell).
node_env_load() {
    command -v node >/dev/null 2>&1 || \
        { [ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 || true; }
    command -v pnpm >/dev/null 2>&1 && return 0
    WEB_SKIP_REASON="node/pnpm not found (install nvm + pnpm — see README)"
    return 1
}

# Soft-skip gate for the verify targets. Exits the calling script: 0 on a loud
# skip, or 1 under STRICT_WEB.
web_guard() {
    node_env_load && [ -d "$WEB_DIR/node_modules" ] && return 0
    echo "SKIPPED: $1 — ${WEB_SKIP_REASON:-JS deps not installed (make install-web)}"
    [ -n "${STRICT_WEB:-}" ] && exit 1
    exit 0
}
