#!/usr/bin/env bash
# Run the interactive Yu-Gi-Oh! client.
# All arguments are forwarded to play_client.py.
#
# Usage:
#   scripts/play.sh                        # interactive mode
#   scripts/play.sh --mode random --seed 42
#   scripts/play.sh --mode greedy --episodes 10 --quiet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_DIR/.venv/bin/activate"
exec python "$PROJECT_DIR/cli/play_client.py" "$@"
