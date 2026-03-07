#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Initializing git submodules..."
git submodule update --init --recursive

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing package with dev dependencies..."
pip install -e ".[dev]"

echo "==> Building libocgcore..."
make build

echo ""
echo "Setup complete! Activate the venv with:"
echo "  source .venv/bin/activate"
echo ""
echo "Note: Some tests require assets/cards.cdb (not included)."
echo "Download it from a ygopro-compatible source and place it at assets/cards.cdb"
