#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

# Parse optional flags
opponent=""
opponent_device=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --opponent)
            opponent="$2"
            shift 2
            ;;
        --opponent-device)
            opponent_device="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--opponent random|greedy|model:PATH] [--opponent-device cpu|cuda]" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$opponent" ]]; then
    export YUGIOH_OPPONENT="$opponent"
fi
if [[ -n "$opponent_device" ]]; then
    export YUGIOH_OPPONENT_DEVICE="$opponent_device"
fi

exec uvicorn yugioh_env.server.app:app --host 0.0.0.0 --port "${PORT:-8000}"
