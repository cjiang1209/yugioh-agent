#!/usr/bin/env bash
# Run the MUD bot client.
# All arguments are forwarded to cli/mud_bot.py.
#
# Usage:
#   scripts/mud_bot.sh                                    # host bot (Player1, creates room)
#   scripts/mud_bot.sh --profile guest                    # guest bot (Player2, joins room)
#   scripts/mud_bot.sh --port 9090 --deck blue_eyes       # custom port and deck
#   scripts/mud_bot.sh --verbose                          # log all protocol lines
#
# Model mode:
#   scripts/mud_bot.sh --mode model:path/to/checkpoint.pt --verbose
#   scripts/mud_bot.sh --mode model:path/to/checkpoint.pt --device cuda
#
# Two-bot duel (run in separate terminals):
#   scripts/mud_bot.sh --profile host --verbose
#   scripts/mud_bot.sh --profile guest --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_DIR/.venv/bin/activate"
exec python "$PROJECT_DIR/cli/mud_bot.py" "$@"
