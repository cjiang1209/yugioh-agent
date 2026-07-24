#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_DIR/.venv/bin/activate"

# Default the mlflow tracking server (only used when --log-to includes mlflow);
# still overridable by exporting MLFLOW_TRACKING_URI before invoking.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000/}"

exec env PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m cli.train "$@"
