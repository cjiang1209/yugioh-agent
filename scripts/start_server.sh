#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

# Parse optional flags
opponent_type=""
checkpoint_path=""
opponent_device=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --opponent)
            opponent_type="$2"
            shift 2
            ;;
        --opponent-checkpoint)
            checkpoint_path="$2"
            shift 2
            ;;
        --opponent-device)
            opponent_device="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--opponent random|greedy|model] [--opponent-checkpoint PATH] [--opponent-device cpu|cuda]" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$opponent_type" ]]; then
    export YUGIOH_OPPONENT_TYPE="$opponent_type"
fi
if [[ -n "$checkpoint_path" ]]; then
    export YUGIOH_OPPONENT_CHECKPOINT="$checkpoint_path"
fi
if [[ -n "$opponent_device" ]]; then
    export YUGIOH_OPPONENT_DEVICE="$opponent_device"
fi

exec uvicorn yugioh_env.server.app:app --host 0.0.0.0 --port "${PORT:-8000}"
