#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MUD_DIR="$PROJECT_ROOT/third_party/yugioh-game"

if [ ! -d "$MUD_DIR" ]; then
    echo "Nothing to clean (third_party/yugioh-game/ does not exist)."
    exit 0
fi

echo "Removing third_party/yugioh-game/ ..."
rm -rf "$MUD_DIR"
echo "Done."
