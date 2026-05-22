#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CORE_DIR="$PROJECT_ROOT/third_party/ygopro-core"
LUA_DIR="$CORE_DIR/lua"
BUILD_DIR="$PROJECT_ROOT/build"

if [ ! -d "$CORE_DIR" ]; then
    echo "Error: ygopro-core not found at $CORE_DIR"
    echo "Run: git submodule add https://github.com/edo9300/ygopro-core.git third_party/ygopro-core"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# Detect OS
UNAME="$(uname -s)"
case "$UNAME" in
    Darwin*)
        SHARED_EXT="dylib"
        SHARED_FLAG="-dynamiclib"
        ;;
    Linux*)
        SHARED_EXT="so"
        SHARED_FLAG="-shared"
        ;;
    *)
        echo "Unsupported OS: $UNAME"
        exit 1
        ;;
esac

OUTPUT="$BUILD_DIR/libocgcore.$SHARED_EXT"

# ─── Detect C++ compiler ────────────────────────────────────────────────────
if [ -z "${CXX:-}" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX=clang++
    elif command -v g++ >/dev/null 2>&1; then
        CXX=g++
    else
        echo "Error: no C++ compiler found (tried clang++, g++)"
        exit 1
    fi
fi
echo "Using C++ compiler: $CXX"

# ─── Build embedded Lua as static library ────────────────────────────────────
LUA_SRC_DIR="$LUA_DIR/src"
if [ ! -d "$LUA_SRC_DIR" ] || [ -z "$(ls "$LUA_SRC_DIR"/*.c 2>/dev/null)" ]; then
    echo "Error: Lua source not found at $LUA_SRC_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Excluded Lua files (matching premake5.lua removefiles)
LUA_EXCLUDE="lbitlib lcorolib ldblib linit loadlib loslib ltests lua luac lutf8lib onelua"

echo "Building embedded Lua (compiled as C++ to match ygopro-core linkage) ..."
LUA_OBJS=""
for src in "$LUA_SRC_DIR"/*.c; do
    base=$(basename "$src" .c)
    # Skip excluded files
    skip=0
    for excl in $LUA_EXCLUDE; do
        if [ "$base" = "$excl" ]; then
            skip=1
            break
        fi
    done
    if [ "$skip" = "1" ]; then
        continue
    fi
    obj="$BUILD_DIR/lua_${base}.o"
    # Compile .c as C++ (compileas "C++" in premake) so symbols have C++ linkage
    $CXX -x c++ -std=c++17 -O2 -fPIC \
        -I"$LUA_SRC_DIR" \
        -I"$LUA_DIR" \
        -include "$LUA_DIR/luaconf-customize.h" \
        -c "$src" -o "$obj"
    LUA_OBJS="$LUA_OBJS $obj"
done
ar rcs "$BUILD_DIR/liblua.a" $LUA_OBJS
echo "Built: $BUILD_DIR/liblua.a"

LUA_CFLAGS="-I$LUA_SRC_DIR -I$LUA_DIR -include $LUA_DIR/luaconf-customize.h"
LUA_LIBS="$BUILD_DIR/liblua.a"

# ─── Find SQLite ─────────────────────────────────────────────────────────────
SQLITE_CFLAGS=""
SQLITE_LIBS=""
if pkg-config --exists sqlite3 2>/dev/null; then
    SQLITE_CFLAGS=$(pkg-config --cflags sqlite3)
    SQLITE_LIBS=$(pkg-config --libs sqlite3)
else
    # Try Xcode SDK
    SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null || echo "")
    if [ -n "$SDK_PATH" ] && [ -f "$SDK_PATH/usr/include/sqlite3.h" ]; then
        SQLITE_CFLAGS="-isysroot $SDK_PATH"
        SQLITE_LIBS="-lsqlite3"
    else
        SQLITE_LIBS="-lsqlite3"
    fi
fi

# ─── Build ygopro-core ───────────────────────────────────────────────────────
SOURCES=$(find "$CORE_DIR" -maxdepth 1 -name '*.cpp' | sort)
RNG_SOURCES=$(find "$CORE_DIR/RNG" -name '*.cpp' 2>/dev/null | sort)
ALL_SOURCES="$SOURCES $RNG_SOURCES"

echo "Building libocgcore.$SHARED_EXT ..."
echo "Sources: $(echo "$SOURCES" | wc -l | tr -d ' ') core + $(echo "$RNG_SOURCES" | wc -l | tr -d ' ') RNG files"

$CXX -std=c++17 -O2 -fPIC \
    $SHARED_FLAG \
    -DOCGCORE_EXPORT_FUNCTIONS \
    -I"$CORE_DIR" \
    $LUA_CFLAGS \
    $SQLITE_CFLAGS \
    $ALL_SOURCES \
    $LUA_LIBS \
    $SQLITE_LIBS \
    -o "$OUTPUT"

echo "Built: $OUTPUT"
echo "Done!"
