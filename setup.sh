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

echo "==> Installing training dependencies (torch, tensorboard)..."
pip install -e ".[train]"

echo "==> Installing embedding dependencies (sentence-transformers)..."
pip install -e ".[embed]"

echo "==> Building libocgcore..."
make build

echo ""

# ── cards.cdb download ──────────────────────────────────────────────────────
CDB_PATH="assets/cards.cdb"
NEED_CDB=false
if [ ! -f "$CDB_PATH" ] || [ ! -s "$CDB_PATH" ]; then
    NEED_CDB=true
fi

if [ "$NEED_CDB" = true ]; then
    echo "The card database (assets/cards.cdb) is missing or empty."
    echo "It is required for running duels and most tests."
    echo ""
    printf "Download cards.cdb automatically? [Y/n] "
    read -r REPLY
    case "$REPLY" in
        [nN]|[nN][oO])
            echo ""
            echo "Skipping download. You can get it later by running:"
            echo "  curl -L -o assets/cards.cdb \\"
            echo "    https://github.com/mycard/ygopro-database/raw/master/locales/en-US/cards.cdb"
            ;;
        *)
            echo "==> Downloading cards.cdb..."
            mkdir -p assets
            curl -L -o "$CDB_PATH" \
                "https://github.com/mycard/ygopro-database/raw/master/locales/en-US/cards.cdb"
            echo "Downloaded $(wc -c < "$CDB_PATH" | tr -d ' ') bytes to $CDB_PATH"
            ;;
    esac
else
    echo "Card database found at $CDB_PATH"
fi

echo ""
echo "Setup complete! Activate the venv with:"
echo "  source .venv/bin/activate"
