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
if [ ! -f "$CDB_PATH" ] || [ ! -s "$CDB_PATH" ]; then
    echo "The card database ($CDB_PATH) is missing or empty."
    echo "It is required for running duels and most tests."
    echo ""
    printf "Download cards.cdb automatically? [Y/n] "
    read -r REPLY
    case "$REPLY" in
        [nN]|[nN][oO])
            echo ""
            echo "Skipping download. You can get it later by running:"
            echo "  curl -fsSL -o $CDB_PATH \\"
            echo "    https://github.com/ProjectIgnis/BabelCDB/raw/master/cards.cdb"
            ;;
        *)
            echo "==> Downloading cards.cdb..."
            mkdir -p assets
            if curl -fsSL -o "$CDB_PATH" \
                "https://github.com/ProjectIgnis/BabelCDB/raw/master/cards.cdb"; then
                echo "    Downloaded $(wc -c < "$CDB_PATH" | tr -d ' ') bytes to $CDB_PATH."
            else
                echo "    Download failed; duels and most tests will not work."
                rm -f "$CDB_PATH"
            fi
            ;;
    esac
else
    echo "Card database found at $CDB_PATH"
fi

# ── strings.conf download (sysstring labels for the string resolver) ────────
STRINGS_PATH="assets/strings.conf"
if [ ! -f "$STRINGS_PATH" ] || [ ! -s "$STRINGS_PATH" ]; then
    echo "==> Downloading strings.conf..."
    mkdir -p assets
    if curl -fsSL -o "$STRINGS_PATH" \
        "https://raw.githubusercontent.com/ProjectIgnis/Distribution/master/config/strings.conf"; then
        echo "    Downloaded $(wc -l < "$STRINGS_PATH" | tr -d ' ') lines to $STRINGS_PATH."
    else
        echo "    Download failed; effect labels will use placeholders."
        rm -f "$STRINGS_PATH"
    fi
else
    echo "Sysstring labels found at $STRINGS_PATH"
fi

echo ""
echo "Setup complete! Activate the venv with:"
echo "  source .venv/bin/activate"
