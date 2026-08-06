#!/usr/bin/env bash
# Download cards.cdb and strings.conf. Never prompts; a failed download is fatal.
set -euo pipefail
cd "$(dirname "$0")/.."

force=""
[ "${1:-}" = "--force" ] && force=1

CDB_URL="https://github.com/ProjectIgnis/BabelCDB/raw/master/cards.cdb"
STRINGS_URL="https://raw.githubusercontent.com/ProjectIgnis/Distribution/master/config/strings.conf"

mkdir -p assets

# Download to a temp file, then move into place, so a failed download can't
# replace a working file.
get() {
    local url="$1" dest="$2" tmp
    tmp="$(mktemp "${dest}.XXXXXX")"
    curl -fsSL -o "$tmp" "$url" || { rm -f "$tmp"; echo "download failed: $url" >&2; exit 1; }
    mv "$tmp" "$dest"
}

if [ -z "$force" ] && [ -s assets/cards.cdb ]; then
    echo "cards.cdb present (FORCE=1 to refresh)"
else
    echo "Downloading cards.cdb ..."; get "$CDB_URL" assets/cards.cdb
fi

if [ -z "$force" ] && [ -s assets/strings.conf ]; then
    echo "strings.conf present (FORCE=1 to refresh)"
else
    echo "Downloading strings.conf ..."; get "$STRINGS_URL" assets/strings.conf
fi
