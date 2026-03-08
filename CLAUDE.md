# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yu-Gi-Oh! RL environment wrapping the C++ ygopro-core engine (edo9300 fork) via ctypes, exposed as a Python API conforming to the `openenv-core` framework. An agent interacts with the environment over HTTP using a FastAPI server.

## Build & Development Commands

Use a virtual environment for all Python commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Note: The Makefile uses `python` (not `python3`). Always activate the venv first, or run commands directly with the venv Python.

```bash
make build      # Compile libocgcore shared library (runs scripts/build_core.sh)
make install    # pip install -e ".[dev]"
make test       # python -m pytest tests/ -v
make clean      # rm -rf build/

# Run a single test file
python -m pytest tests/test_message_parser.py -v

# Run a specific test
python -m pytest tests/test_duel.py::test_name -v

# Run the server
uvicorn yugioh_env.server.app:app --host 0.0.0.0 --port 8000

# Run the interactive play client (server must be running)
scripts/play_client.sh                        # interactive mode
scripts/play_client.sh --mode random --seed 42
scripts/play_client.sh --mode greedy --episodes 10 --quiet
scripts/play_client.sh --deck assets/decks/blue_eyes.ydk --mode greedy
scripts/play_client.sh --deck0 assets/decks/blue_eyes.ydk --deck1 assets/decks/starter.ydk
```

## Prerequisites & Setup

- `clang++` with C++17, `sqlite3` headers
- Git submodules for ygopro-core and CardScripts:
  ```bash
  git submodule update --init --recursive
  ```
- `assets/cards.cdb` — SQLite card database (not included, must be downloaded separately)
- Python 3.10+

## Test Skip Behavior

Tests auto-skip when prerequisites are missing:
- `lib` fixture: skips if `libocgcore` not built (`make build`)
- `db_path` fixture: skips if `assets/cards.cdb` absent
- `script_dirs` fixture: skips if `third_party/CardScripts/` absent
- Pure unit tests (`test_message_parser`, `test_observation`, `test_action_space`, `test_deck_parser`) require no external deps

## Architecture

```
HTTP Client (YuGiOhEnv)
        ↕  openenv-core HTTP
FastAPI Server (server/app.py)
        ↕
YuGiOhEnvironment (server/yugioh_environment.py)
  ├── Duel (duel.py)             — manages one duel lifetime via OCG C API
  │     ├── DuelCallbacks        — bridges Python ↔ C (card data, script loading)
  │     ├── CardDatabase         — reads card stats from cards.cdb via sqlite3
  │     ├── GameState            — tracks LP, zones, phase, turn from parsed messages
  │     └── libocgcore (ctypes)  — C++ engine loaded via lib_loader.py
  ├── ActionMapper               — maps SELECT messages to fixed action space (32 max)
  ├── Opponent                   — auto-plays opponent (Random or Greedy)
  └── build_observation()        — encodes GameState into numpy arrays for RL
```

### Data Flow Per Step

1. `step(action)` receives `YuGiOhAction(action_index=N)`
2. `ActionMapper.action_to_response(N)` converts to binary buffer
3. `Duel.send_response()` calls `OCG_DuelSetResponse` on the C engine
4. `Duel.process_until_choice()` loops: `OCG_DuelProcess` → `OCG_DuelGetMessage` → `parse_messages()` → `GameState.update()`, returning on agent SELECT messages, auto-playing opponent messages, or game end
5. `build_observation()` queries zones via `OCG_DuelQueryLocation`, returns numpy arrays

### Observation Space

- `cards`: `(200, 42)` uint8 — up to 200 cards, 42 features each
- `global_state`: `(20,)` uint8 — LP, phase, turn, zone counts
- `actions`: `(32, 12)` uint8 — per-action feature vectors
- `action_mask`: `(32,)` int8 — 1=legal, 0=illegal

### Action Space

Fixed 32 actions (`MAX_ACTIONS`). `ActionMapper` handles 21+ `MSG_SELECT_*` types (idle cmd, battle cmd, card selection, tribute, chain, place, position, sum, yes/no, etc.).

### Client-Specified Decks

`reset()` accepts optional `deck0`/`deck1` inline deck dicts (`{"main": [int, ...], "extra": [int, ...]}`). When omitted, the server-configured default deck paths are used. Validation enforces 40-60 main cards, 0-15 extra cards, all positive ints. The play client supports `--deck`, `--deck0`, `--deck1` flags (paths to `.ydk` files parsed client-side).

## Critical Implementation Details

1. **ctypes GC safety**: `DuelCallbacks` stores all ctypes arrays (setcode arrays, script content) as instance attributes to prevent Python GC from freeing memory still referenced by the C engine. Do not remove these storage dicts.

2. **Lua compiled as C++**: The embedded Lua sources are compiled with `clang++ -x c++` to match ygopro-core's C++ symbol linkage. This must not be changed to plain C compilation.

3. **Message framing**: The edo9300 fork prefixes each message buffer entry with a 4-byte `uint32 LE` length field, differing from original ygopro protocol.

4. **Thread safety**: `YuGiOhEnvironment.SUPPORTS_CONCURRENT_SESSIONS = False`. One instance must not run concurrent sessions.

5. **Deck shuffling**: The engine's `Startup` processor clears shuffle flags before the opening draw, so `Duel._add_deck_cards()` shuffles the main deck in Python (using a seeded `random.Random`) before inserting cards. Extra deck cards must be listed under `#extra` in `.ydk` files; cards in the `main` list are always added to the main deck.

6. **Seed handling**: Seeds are spread across 4 `uint64` slots (xoshiro256** RNG) via LCG mixing. Zero seeds are mapped to 1 (engine requires non-zero).

## Environment Variables

- `YUGIOH_LIB_PATH` — path to `libocgcore.dylib/.so` (auto-detected from `build/` if unset)
- `YUGIOH_DB_PATH` — path to `cards.cdb` (default: `assets/cards.cdb`)
