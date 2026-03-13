#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_DIR/.venv/bin/activate"

LOGDIR="checkpoints"
PORT="${TENSORBOARD_PORT:-6006}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logdir) LOGDIR="$2"; shift 2 ;;
        --logdir=*) LOGDIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--logdir DIR]"; exit 1 ;;
    esac
done

echo "Starting TensorBoard on port $PORT (logdir: $LOGDIR)"
echo "Opening http://localhost:$PORT ..."

# Open browser after a short delay to let the server start
(sleep 2 && open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null) &

exec tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
