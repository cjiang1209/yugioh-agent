#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MUD_DIR="$PROJECT_ROOT/third_party/yugioh-game"

if [ ! -d "$MUD_DIR" ] || [ ! -d "$MUD_DIR/.venv" ]; then
    echo "Error: MUD server not built."
    echo "Run: scripts/build_mud_server.sh"
    exit 1
fi

source "$MUD_DIR/.venv/bin/activate"
exec python "$PROJECT_ROOT/cli/seed_mud.py" "$@"
