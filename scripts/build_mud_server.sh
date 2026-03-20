#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MUD_DIR="$PROJECT_ROOT/third_party/yugioh-game"

# Detect OS
UNAME="$(uname -s)"
case "$UNAME" in
    Darwin*)
        LUA_TARGET="macosx"
        ;;
    Linux*)
        LUA_TARGET="linux"
        ;;
    *)
        echo "Unsupported OS: $UNAME"
        exit 1
        ;;
esac

# ─── 1. Clone yugioh-game ───────────────────────────────────────────────────
if [ ! -d "$MUD_DIR" ]; then
    echo "Cloning yugioh-game ..."
    git clone https://github.com/tspivey/yugioh-game.git "$MUD_DIR"
else
    echo "yugioh-game already cloned, skipping."
fi

# ─── 2. Download & compile Lua 5.3.5 ────────────────────────────────────────
LUA_DIR="$MUD_DIR/lua-5.3.5"
if [ ! -f "$LUA_DIR/src/liblua.a" ]; then
    echo "Building Lua 5.3.5 ..."
    cd "$MUD_DIR"
    if [ ! -d "$LUA_DIR" ]; then
        curl -LO https://www.lua.org/ftp/lua-5.3.5.tar.gz
        tar xzf lua-5.3.5.tar.gz
        rm -f lua-5.3.5.tar.gz
    fi
    cd "$LUA_DIR"
    make "$LUA_TARGET" CC=clang++ CFLAGS='-O2 -fPIC'
    echo "Built: $LUA_DIR/src/liblua.a"
else
    echo "Lua 5.3.5 already built, skipping."
fi

# ─── 3. Clone & patch Fluorohydride/ygopro-core ─────────────────────────────
# Pin to last commit before int32→int32_t typedef change (a6ddb76), since
# duel_build.py's inline C code uses the old typedefs (int32, uint32, uint8).
CORE_DIR="$MUD_DIR/ygopro-core"
CORE_PIN="6871274"
if [ ! -d "$CORE_DIR" ]; then
    echo "Cloning Fluorohydride/ygopro-core (pinned to $CORE_PIN) ..."
    git clone https://github.com/Fluorohydride/ygopro-core.git "$CORE_DIR"
    cd "$CORE_DIR" && git checkout "$CORE_PIN"
else
    echo "ygopro-core already cloned, skipping."
fi

# Apply patch: remove 'static' from is_declarable() so it has external linkage.
# The shipped patch (etc/ygopro-core.patch) targets an older API (int32 vs int32_t),
# so we apply the fix directly with sed for robustness.
if grep -q '^static.*is_declarable' "$CORE_DIR/playerop.cpp" 2>/dev/null; then
    echo "Patching playerop.cpp (removing static from is_declarable) ..."
    sed -i.bak 's/^static \(.*is_declarable\)/\1/' "$CORE_DIR/playerop.cpp"
    rm -f "$CORE_DIR/playerop.cpp.bak"
else
    echo "playerop.cpp already patched or not found, skipping."
fi

# ─── 4. Compile libygo.so ───────────────────────────────────────────────────
# Always .so, even on macOS (.dylib convention). CFFI's duel_build.py hardcodes
# libraries=['ygo'] which resolves to libygo.so, and dlopen handles .so on macOS.
if [ ! -f "$MUD_DIR/libygo.so" ]; then
    echo "Building libygo.so ..."
    cd "$CORE_DIR"
    clang++ -shared -fPIC -o "$MUD_DIR/libygo.so" *.cpp \
        -I"$LUA_DIR/src" -L"$LUA_DIR/src" -llua -std=c++14
    echo "Built: $MUD_DIR/libygo.so"
else
    echo "libygo.so already built, skipping."
fi

# ─── 5. Clone ygopro-scripts ────────────────────────────────────────────────
SCRIPT_DIR_MUD="$MUD_DIR/script"
if [ ! -d "$SCRIPT_DIR_MUD" ]; then
    echo "Cloning ygopro-scripts ..."
    git clone https://github.com/Fluorohydride/ygopro-scripts.git "$SCRIPT_DIR_MUD"
else
    echo "ygopro-scripts already cloned, skipping."
fi

# ─── 6. Create venv & install deps ──────────────────────────────────────────
VENV_DIR="$MUD_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Check if deps are installed by looking for twisted
if ! python -c "import twisted" 2>/dev/null; then
    echo "Installing dependencies ..."
    # The pinned versions (Twisted 18.4.0, etc.) are too old for Python 3.10+.
    # Install without version pins — the MUD server uses standard Twisted APIs.
    # Pin sqlalchemy<2 (MUD server uses deprecated MetaData.bind removed in 2.0).
    # Pin attrs<24 (newer attrs breaks gsb's attribute ordering).
    pip install \
        setuptools \
        twisted \
        cffi \
        autobahn \
        natsort \
        passlib \
        pyopenssl \
        'sqlalchemy>=1.4,<2' \
        alembic \
        Babel \
        MailChecker \
        'attrs>=22.2,<24' \
        "git+https://github.com/chrisnorman7/game-server-base.git" \
        || {
            echo ""
            echo "Error: pip install failed."
            exit 1
        }
    # Patch gsb for Python 3.12+: re._pattern_type was removed in 3.7.
    GSB_DIR="$(pip show gsb 2>/dev/null | grep Location | cut -d' ' -f2)/gsb"
    GSB_COMMAND="$GSB_DIR/command.py"
    if grep -q "_pattern_type" "$GSB_COMMAND" 2>/dev/null; then
        echo "Patching gsb for Python 3.12+ compatibility ..."
        sed -i.bak \
            -e 's/from re import compile, _pattern_type/from re import compile, Pattern/' \
            -e 's/_pattern_type/Pattern/g' \
            "$GSB_COMMAND"
        rm -f "$GSB_COMMAND.bak"
    fi

    # Patch gsb Reader for attrs>=22 attribute ordering.
    # In newer attrs, Reader.done (overridden with a default) moves to the end
    # of the attribute list. When code passes done as the first positional arg,
    # it gets assigned to command_separator instead. Fix via __attrs_post_init__.
    GSB_INTERCEPT="$GSB_DIR/intercept.py"
    # Guard: check specifically for the Reader __attrs_post_init__ (Menu already has one)
    if ! grep -q 'callable(self.command_separator)' "$GSB_INTERCEPT" 2>/dev/null; then
        echo "Patching gsb Reader for attrs>=22 attribute ordering ..."
        sed -i.bak '/^    done=attrib(default=Factory(lambda: None))/a\
\
    def __attrs_post_init__(self):\
        """Fix attrs ordering: if done is None but command_separator is\
        callable, the first positional arg was misrouted."""\
        if self.done is None and callable(self.command_separator):\
            self.done = self.command_separator\
            self.command_separator = '"'"' '"'"'' \
            "$GSB_INTERCEPT"
        rm -f "$GSB_INTERCEPT.bak"
    fi
else
    echo "Python dependencies already installed, skipping."
fi

# ─── 7a. Patch duel.py for Python 3.12+: find_module/load_module removed ──
# pkgutil.iter_modules returns FileFinder objects that no longer have
# find_module() in Python 3.12+. Replace with importlib.import_module().
DUEL_PY="$MUD_DIR/ygo/duel.py"
if grep -q 'importer.find_module' "$DUEL_PY" 2>/dev/null; then
    echo "Patching duel.py for Python 3.12+ (importlib) ..."
    sed -i.bak \
        -e '1,/^import os/{/^import os/a\
import importlib
}' \
        -e "s|m = importer.find_module(modname).load_module(modname)|m = importlib.import_module(f'.message_handlers.{modname}', package='ygo')|" \
        "$DUEL_PY"
    rm -f "$DUEL_PY.bak"
else
    echo "duel.py already patched, skipping."
fi

# ─── 7b. Patch duel_build.py for local paths ────────────────────────────────
# The upstream duel_build.py assumes:
#   - ygopro-core is at ../ygopro-core (parent dir), but we clone it into ./ygopro-core
#   - Lua headers at /usr/include/lua5.3 (system Lua on Linux), but we build locally
#   - -std=c++0x (C++11), but current ygopro-core needs C++14 (std::exchange)
if grep -q "'/usr/include/lua5.3'" "$MUD_DIR/duel_build.py" 2>/dev/null; then
    echo "Patching duel_build.py for local include paths ..."
    sed -i.bak \
        -e "s|'../ygopro-core'|'./ygopro-core'|" \
        -e "s|'./core', '/usr/include/lua5.3'|'./lua-5.3.5/src'|" \
        -e "s|-std=c++0x|-std=c++14|" \
        "$MUD_DIR/duel_build.py"
    # On macOS, Python C extensions need -undefined dynamic_lookup for symbol resolution
    if [ "$UNAME" = "Darwin" ]; then
        sed -i.bak \
            "s|extra_link_args=\['-Wl,-rpath,.'\]|extra_link_args=['-Wl,-rpath,.', '-undefined', 'dynamic_lookup']|" \
            "$MUD_DIR/duel_build.py"
    fi
    rm -f "$MUD_DIR/duel_build.py.bak"
else
    echo "duel_build.py already patched, skipping."
fi

# ─── 8. Run CFFI build ──────────────────────────────────────────────────────
if ! ls "$MUD_DIR"/_duel*.so 1>/dev/null 2>&1; then
    echo "Running CFFI duel_build.py ..."
    cd "$MUD_DIR"
    python duel_build.py
    echo "Built CFFI duel module."
else
    echo "CFFI duel module already built, skipping."
fi

# ─── 9. Symlink cards.cdb ───────────────────────────────────────────────────
LOCALE_DIR="$MUD_DIR/locale/en"
mkdir -p "$LOCALE_DIR"
if [ -f "$PROJECT_ROOT/assets/cards.cdb" ]; then
    ln -sf "$PROJECT_ROOT/assets/cards.cdb" "$LOCALE_DIR/cards.cdb"
    echo "Symlinked cards.cdb into locale/en/."
else
    echo "Warning: assets/cards.cdb not found. You will need to provide it before running the server."
fi

# ─── 10. Run alembic migrations (only if game.db already exists) ──────────────
# On a fresh build there is no game.db yet — the server creates it on first
# startup via Base.metadata.create_all() with the latest schema, so no
# migrations are needed.  Only run alembic on an existing DB.
cd "$MUD_DIR"
if [ -f "$MUD_DIR/game.db" ]; then
    echo "Running alembic migrations on existing game.db ..."
    alembic upgrade head || echo "Warning: alembic migration failed (non-fatal)."
else
    echo "No game.db found, skipping alembic migrations (created on first server start)."
fi

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "=== MUD server build complete ==="
echo "Run: scripts/start_mud_server.sh"
echo "Connect: telnet localhost 4000"
