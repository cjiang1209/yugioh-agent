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

Note: `make test` activates the venv automatically. For other Makefile targets and manual Python commands, activate the venv first.

```bash
make build      # Compile libocgcore shared library (runs scripts/build_core.sh)
make install    # pip install -e ".[dev]"
make test       # python -m pytest tests/ -v
make clean      # rm -rf build/

# Run a single test file
python -m pytest tests/env/test_message_parser.py -v

# Run a specific test
python -m pytest tests/env/test_duel.py::test_name -v

# Run only one module's tests
python -m pytest tests/mud/ -v

# Run the server (activates venv automatically)
scripts/start_server.sh
scripts/start_server.sh --opponent model:checkpoints/latest.pt

# Run the interactive play client (server must be running)
scripts/play_client.sh                        # interactive mode
scripts/play_client.sh --mode random --seed 42
scripts/play_client.sh --mode greedy --episodes 10 --quiet
scripts/play_client.sh --deck assets/decks/blue_eyes.ydk --mode greedy
scripts/play_client.sh --deck0 assets/decks/blue_eyes.ydk --deck1 assets/decks/starter.ydk
scripts/play_client.sh --go-second --mode greedy          # agent goes second (player 1)

# Run the MUD bot client (MUD server must be running)
pip install -e ".[mud]"                                    # adds websockets
scripts/mud_bot.sh                                         # host bot (Player1, creates room)
scripts/mud_bot.sh --profile guest                         # guest bot (Player2, joins room)
scripts/mud_bot.sh --port 9090 --deck blue_eyes            # custom port and deck
scripts/mud_bot.sh --verbose                               # log all protocol lines
# Two-bot duel (run in separate terminals):
#   scripts/mud_bot.sh --profile host --verbose
#   scripts/mud_bot.sh --profile guest --verbose
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

Tests are organized into subdirectories by module (`tests/core/`, `tests/env/`, `tests/mud/`, `tests/rl/`, `tests/cli/`). Run a single module's tests with e.g. `python -m pytest tests/mud/ -v`.

Tests auto-skip when prerequisites are missing:
- `lib` fixture (`tests/env/conftest.py`): skips if `libocgcore` not built (`make build`)
- `db_path` fixture (`tests/conftest.py`): skips if `assets/cards.cdb` absent
- `script_dirs` fixture (`tests/env/conftest.py`): skips if `third_party/CardScripts/` absent
- Pure unit tests (`tests/env/test_message_parser`, `tests/env/test_observation`, `tests/env/test_action_space`, `tests/env/test_deck_parser`) require no external deps
- `tests/env/test_opponent.py` ModelOpponent tests: skips if `torch` not installed
- `tests/rl/test_card_embeddings.py`: TextEmbeddingLookup/network tests skip if `torch` not installed; `test_build_embeddings_output_structure` skips if `sentence-transformers` not installed
- `tests/rl/test_checkpoint_init.py`: skips if `torch` not installed
- `tests/rl/test_eval_opponents.py`: skips if `torch` not installed
- `tests/rl/test_resume.py`: skips if `torch` not installed
- `tests/mud/test_protocol.py`: pure unit tests (uses FakeConnection), no external deps
- `tests/mud/test_text_parser.py`: pure unit tests for duel prompt/event parsing, no external deps
- `tests/mud/test_card_lookup.py`: pure unit tests (uses temp SQLite DB), no external deps
- `tests/mud/test_game_state.py`: pure unit tests (uses temp SQLite DB for CardNameLookup), no external deps
- `tests/mud/test_observation.py`: pure unit tests (uses temp SQLite DB for CardDatabase), no external deps
- `tests/mud/test_model_agent.py`: skips if `torch` not installed

## Architecture

```
yugioh_core  (zero project deps — shared primitives)
  ├── constants.py         — ygopro-core constants, split_setcodes(), PHASE_NAMES, SELECT_MSGS
  ├── encoding.py          — observation encoding (encode_card, encode_u16/u32, dims, ZONE_SLOTS)
  ├── card_database.py     — CardDatabase (SQLite .cdb reader)
  └── action_categories.py — named idle/battle action category constants (IDLE_SUMMON, etc.)
     ↑
  ┌──┴──┐
yugioh_env  yugioh_mud  (both import core; mud does not depend on env)
     ↑         ↑
     └── yugioh_rl (lazy imports for ModelAgent/ModelOpponent)

HTTP Client (YuGiOhEnv)
        ↕  openenv-core HTTP
FastAPI Server (server/app.py)
        ↕
YuGiOhEnvironment (server/yugioh_environment.py)
  ├── Duel (duel.py)             — manages one duel lifetime via OCG C API
  │     ├── DuelCallbacks        — bridges Python ↔ C (card data, script loading)
  │     ├── CardDatabase         — reads card stats from cards.cdb (from yugioh_core)
  │     ├── GameState            — tracks LP, zones, phase, turn from parsed messages
  │     └── libocgcore (ctypes)  — C++ engine loaded via lib_loader.py
  ├── ActionMapper               — maps SELECT messages to fixed action space (32 max)
  ├── Opponent                   — auto-plays opponent (Random, Greedy, or Model)
  └── build_observation()        — encodes GameState into numpy arrays for RL
```

### Data Flow Per Step

1. `step(action)` receives `YuGiOhAction(action_index=N)`
2. `ActionMapper.action_to_response(N)` converts to binary buffer
3. `Duel.send_response()` calls `OCG_DuelSetResponse` on the C engine
4. `Duel.process_until_choice()` loops: `OCG_DuelProcess` → `OCG_DuelGetMessage` → `parse_messages()` → `GameState.update()`, returning on agent SELECT messages, auto-playing opponent messages, or game end
5. `build_observation()` queries zones via `OCG_DuelQueryLocation`, returns numpy arrays

### Observation Space

- `cards`: `(200, 42)` uint8 — up to 200 cards, 42 bytes each (30 used, 12 padding); decoded to 95 float features + card_id
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

## ygopro-core Wire Format Reference

When adding or modifying message parsers (`message_parser.py`) or response builders (`response_builder.py`), **always verify field types and order against the C++ source** in `third_party/ygopro-core/`. Past bugs have all been mismatches between what C++ writes and what Python reads.

### `loc_info` struct (most common source of bugs)

Defined in `card.h`. Written by `message->write(pcard->get_info_location())`:
```
u8  controler
u8  location
u32 sequence
u32 position
```
Total: **10 bytes**. Use `BinaryReader.read_card_loc()` whenever C++ writes a `loc_info`. Never hand-roll `u8,u8,u8,padding` — that reads only 4 bytes.

Messages that write `loc_info`: `MSG_MOVE`, `MSG_SET`, `MSG_SUMMONING`, `MSG_SPSUMMONING`, `MSG_FLIPSUMMONING`, `MSG_CHAINING`, `MSG_ATTACK` (×2), `MSG_BATTLE` (×2), `MSG_EQUIP` (×2), `MSG_CARD_TARGET` (×2), `MSG_CANCEL_TARGET` (×2), `MSG_CARD_HINT`, `MSG_BECOME_TARGET` (×N), `MSG_RANDOM_SELECTED` (×N), `MSG_CARD_SELECTED` (×N), `MSG_SHUFFLE_SET_CARD` (×2N), `MSG_SELECT_SUM` (×N, before param), `MSG_SELECT_CARD`, `MSG_SELECT_UNSELECT_CARD`, `MSG_SELECT_TRIBUTE`, `MSG_SELECT_EFFECTYN`.

### Common encoding pitfalls

1. **C++ may write a narrow field as a wider type.** For example, `current.location` is `uint8_t` in the struct but `SortCard` writes it via `write<uint32_t>(pcard->current.location)` — 4 bytes on the wire, not 1.

2. **`client_mode` u8 field** appears at the end of activatable card entries in `MSG_SELECT_IDLECMD`, `MSG_SELECT_BATTLECMD`, and `MSG_SELECT_CHAIN`. Easy to miss.

3. **Response buffers have no framing.** The raw bytes are copied into `returns.data` and accessed via `returns.at<T>(index)` where byte offset = `index * sizeof(T)`. No length prefix unless the C++ code explicitly reads one (e.g., `parse_response_cards` reads a type discriminator at position 0).

4. **`parse_response_cards()`** (used by `MSG_SELECT_CARD`, `MSG_SELECT_TRIBUTE`, `MSG_SELECT_SUM`) expects: `int32(type) + uint32(count) + indices...`. Type=0 → uint32 indices, type=1 → uint16, type=2 → uint8, type=3 → bitmask.

5. **`MSG_SELECT_COUNTER` response** has no length prefix — the engine reads `returns.at<int16_t>(i)` directly for `i` in `0..card_count-1`.

### Verification checklist for new parsers

- [ ] Compare field-by-field against the `message->write<T>()` calls in the C++ processor
- [ ] Check if `write(loc_info)` is used — if so, use `read_card_loc()` (10 bytes, not 4)
- [ ] Check if `write<uint32_t>(uint8_field)` widens a narrow value
- [ ] For response builders, check `returns.at<T>(pos)` access patterns in the C++ `step == 1` branch
- [ ] Run `make test` and a short random-play session to verify no MSG_RETRY

## RL Training System (`yugioh_rl/`)

PPO-based training that calls `YuGiOhEnvironment` directly in-process (no HTTP).

### Installation

```bash
pip install -e ".[train]"   # adds torch + tensorboard
pip install -e ".[embed]"  # adds sentence-transformers (for building card text embeddings)
```

### Running Training

```bash
scripts/train.sh                                           # default: 8 envs, 1M steps, greedy opponent
scripts/train.sh --num-envs 4 --total-timesteps 500000     # fewer envs, shorter run
scripts/train.sh --opponent random --no-reward-shaping      # sparse rewards only
scripts/train.sh --opponent model:checkpoints/latest.pt    # self-play
scripts/train.sh --device cuda --base-dir runs/exp1         # GPU + custom base dir
scripts/train.sh --agent-player random                     # coin flip per episode (default)
scripts/train.sh --agent-player first                      # always go first
scripts/train.sh --agent-player second                     # always go second
scripts/train.sh --card-embeddings assets/card_text_embeddings.pt  # text-aware card encoding
scripts/train.sh --init-checkpoint checkpoints/run1/checkpoint_100.pt  # new run from existing weights
scripts/train.sh --init-checkpoint checkpoints/run1/checkpoint_100.pt --resume-optimizer  # also load optimizer state
scripts/train.sh --resume checkpoints/run1/checkpoint_latest.pt                        # resume interrupted run
scripts/train.sh --resume checkpoints/run1/checkpoint_100.pt --total-timesteps 2000000 # resume and extend training
scripts/train.sh --eval-opponents greedy random                        # default eval opponents
scripts/train.sh --eval-opponents greedy model:checkpoints/run1/latest.pt  # eval vs model checkpoint
scripts/train.sh --eval-opponents greedy model:checkpoints/v1/latest.pt model:checkpoints/v2/latest.pt  # multiple models

# Build card text embeddings (requires sentence-transformers)
scripts/build_card_embeddings.sh                           # default: assets/cards.cdb → assets/card_text_embeddings.pt
scripts/build_card_embeddings.sh --db path/to/cards.cdb --output path/to/embeddings.pt
```

See all options: `scripts/train.sh --help`

Each run auto-creates a timestamped subdirectory under `--base-dir` (e.g. `checkpoints/20260311_143000_seed42/`) containing `config.json`, checkpoints, and TensorBoard logs. This prevents runs from overwriting each other.

### Module Layout

```
yugioh_rl/
├── config.py        — TrainingConfig dataclass (all hyperparameters)
├── features.py      — Decode uint8 observations → float tensors for neural net
├── network.py       — YuGiOhNet: card encoder + zone pooling + dot-product policy head + value head
├── env_wrapper.py   — TrainingEnv (single, in-process) + SubprocVecEnv (multiprocessing)
├── ppo.py           — RolloutBuffer + PPOTrainer (GAE, clipped surrogate, eval, checkpoints)
cli/train.py         — CLI entry point (argparse → TrainingConfig → PPOTrainer.train())
scripts/train.sh     — Shell wrapper (activates venv, forwards args)
```

### Network Architecture

- **Card encoder**: two modes — **symbolic** (default: modulo-hashed learned embedding, cards are arbitrary tokens) or **semantic** (`--card-embeddings`: frozen text embeddings + collision-free learned embedding, cards carry meaning from effect text). Followed by MLP per card → zone-pooled board representation.
- **Policy head**: dot-product scoring of action embeddings against board projection, masked by `action_mask`
- **Value head**: MLP on board representation → scalar

### Reward Shaping

Enabled by default. Adds per-step shaping to the sparse terminal reward (+1 win, −1 loss):
- LP delta: `w_lp * (delta_my_lp - delta_opp_lp) / 8000`
- Card advantage delta: `w_card * delta_hand_advantage`

Disable with `--no-reward-shaping`.

### Key Design Decisions

1. **In-process environment**: `TrainingEnv` wraps `YuGiOhEnvironment` directly, bypassing HTTP serialization overhead. Observations are kept as numpy arrays.
2. **Subprocess vectorization**: `SubprocVecEnv` spawns N worker processes (one `TrainingEnv` each) using `multiprocessing.spawn` context, respecting the single-session constraint.
3. **Shared card embedding**: The same `nn.Embedding` encodes board card IDs and action card codes so the network learns a single card representation.
4. **Auto-reset**: `TrainingEnv.step()` auto-resets on episode end, returning the first obs of a new episode and storing terminal info.
5. **Player order randomization**: By default (`--agent-player random`), the agent randomly goes first or second each episode (coin flip seeded by the episode seed). This prevents training bias from always playing first. The observation/network architecture is already player-agnostic (relativized by `agent_player`), so no model changes are needed.
6. **Model opponent (self-play)**: `ModelOpponent` loads a trained checkpoint and runs greedy argmax inference to select actions. When `needs_observation` is True, the environment builds a full observation from the opponent's perspective before each decision. The server supports `--opponent model:PATH` (also configurable via `YUGIOH_OPPONENT` env var).
7. **Semantic card embeddings (optional)**: The network supports two card embedding modes — **symbolic** (default: cards are arbitrary tokens, modulo-hashed into a learned embedding) and **semantic** (`--card-embeddings`: cards carry meaning from effect text). In semantic mode, `TextEmbeddingLookup` loads pre-computed sentence-transformer embeddings and uses `torch.searchsorted` for vectorized lookup by passcode. Frozen text vectors are projected via trainable `nn.Linear` and concatenated with a collision-free learned embedding. The embeddings file lives only in the trainer process — `SubprocVecEnv` workers never load it.
8. **Incremental training from checkpoint**: `--init-checkpoint PATH` starts a new run (fresh directory, counters at 0) with model weights initialized from an existing checkpoint instead of random init. `--resume-optimizer` additionally loads optimizer state (momentum/variance), with LR overridden from the CLI. Architecture dimensions must match between checkpoint and CLI config; `PPOTrainer._validate_checkpoint_compat` checks this at startup. Text embedding mode must also be compatible (cannot add text embeddings to a symbolic checkpoint).
9. **Resume interrupted training**: `--resume PATH` restores full training state (model weights, optimizer, update/step counters, episode tracking) and continues in the same run directory. The `--total-timesteps` CLI value is always recomputed — pass a higher value to extend training or a lower value (triggers early return if already past). `--resume` and `--init-checkpoint` are mutually exclusive. TensorBoard logs continue seamlessly via `purge_step`. **Known limitation — episode seed divergence**: on resume, `SubprocVecEnv` is created with the original `config.seed` and `vec_env.reset()` replays the episode seed sequence from the beginning, not from where the interrupted run left off. Training is unaffected (the model still learns), but the exact episode ordering will differ from a single uninterrupted run. Saving and restoring per-env RNG state is impractical given the multi-process architecture.

## MUD Server (yugioh-game)

The yugioh-game MUD server (tspivey/yugioh-game) is a Twisted-based text MUD for multiplayer Yu-Gi-Oh dueling. It uses a **different** ygopro-core fork (Fluorohydride, not edo9300) with incompatible C APIs, so it has its own build pipeline.

### Build & Run

```bash
scripts/build_mud_server.sh                    # Clone deps, compile Lua 5.3.5 + Fluorohydride/ygopro-core, install Python deps
scripts/start_mud_server.sh                    # Start: telnet on 4000, WebSocket on 8080
scripts/start_mud_server.sh --port 5000        # Custom telnet port
WS_PORT=9090 scripts/start_mud_server.sh       # Custom WebSocket port
scripts/clean_mud_server.sh                    # Remove third_party/yugioh-game/ entirely
scripts/seed_mud_accounts.sh                   # Create Player1/Player2 accounts + load decks from assets/decks/
scripts/seed_mud_accounts.sh --deck path/to/deck.ydk  # Seed specific .ydk files only
telnet localhost 4000                          # Connect via telnet
```

### Key Differences from Main Project

- **ygopro-core fork**: Fluorohydride (C API: `create_duel`, `process`, `set_responsei`) vs edo9300 (`OCG_CreateDuel`, `OCG_DuelProcess`, `OCG_DuelSetResponse`). Completely incompatible APIs.
- **Shared library**: `libygo.so` (CFFI) vs `libocgcore.dylib/.so` (ctypes). No conflict.
- **Separate venv**: `third_party/yugioh-game/.venv` — old pinned deps (Twisted 18.4.0, SQLAlchemy 1.3.4) that conflict with the main project.
- **Cloned on demand**: Not a git submodule. `third_party/yugioh-game/` is gitignored, treated as a build artifact.
- **Lua 5.3.5**: Downloaded and compiled by the build script (compiled as C++ with `CC=clang++` for C++ linkage, same rationale as the main project's Lua build).
- **Python 3.12+ patches** (applied automatically by `build_mud_server.sh`):
  - `duel.py`: `pkgutil.iter_modules()` returns `FileFinder` objects that lack `find_module()` (removed in 3.12). Patched to use `importlib.import_module()`. Without this, all message handlers fail to load silently and duels hang.
  - `gsb/command.py`: `re._pattern_type` was removed in 3.7. Patched to use `re.Pattern`.
  - `gsb/intercept.py`: `Reader.done` attribute reordering under attrs>=22. When `done` is overridden with a default, it moves to the end of the attribute list, causing positional args to be misrouted. Patched with `__attrs_post_init__` to detect and fix the misrouted callable.

### MUD Bot Client (`yugioh_mud/`)

WebSocket bot that connects to the MUD server and plays through complete duels (login → lobby → room setup → RPS → go-first decision → duel → finished).

```
yugioh_mud/
├── config.py              — MUDBotConfig dataclass, HOST_CONFIG / GUEST_CONFIG presets
├── connection.py          — Async WebSocket client wrapper (send_line / recv_line)
├── protocol.py            — State machine: LOGIN → LOBBY → ROOM_SETUP → RPS → DECISION → DUEL → FINISHED
├── text_parser.py         — Line-oriented duel prompt classifier (PromptType enum + ParsedPrompt)
│                            + event parser (EventType enum + ParsedEvent) for informational lines
├── card_lookup.py         — Name-to-passcode reverse index from cards.cdb texts table
├── game_state.py          — Zone tracking (MUDGameState) consuming ParsedEvent + CardNameLookup
├── cmd_handler.py         — Atomic idle/battle handlers (probe→build→decide→execute)
├── observation.py         — MUDGameState → numpy RL observation arrays via encode_card() + CardDatabase
├── agent.py               — Agent protocol + PassiveAgent + RandomAgent + ModelAgent
└── action_translator.py   — Converts agent int actions → MUD text commands
cli/mud_bot.py             — CLI entry point (--profile host/guest, --deck, --verbose)
scripts/mud_bot.sh         — Shell wrapper (activates venv, forwards args)
```

- **Install**: `pip install -e ".[mud]"` (adds `websockets>=12.0`)
- **Profiles**: `host` (Player1, creates room, sends `start`) vs `guest` (Player2, joins host's room). Defaults use accounts seeded by `scripts/seed_mud_accounts.sh`.
- **State machine**: `MUDProtocol` is a line-oriented async state machine. Each state handler pattern-matches server lines and sends commands. The `Connection` protocol interface enables unit testing with a `FakeConnection`.
- **Text parser**: `MUDTextParser` classifies MUD server lines into 21 `PromptType` variants (idle/battle menus, card/tribute/chain selection, effect Y/N, position, place, option, sum, counter, unselect, announce, sort). It tracks idle/battle context and accumulates numbered option lines until a known terminal line arrives. Each prompt type maps to a specific MUD server mechanism (DuelReader with/without prompt, DuelMenu, yes_or_no_parser). Additionally parses 27 `EventType` variants (turns, phases, LP changes, draws, summons, attacks, card movement including GY/banished→field returns, chains, win/lose) into `ParsedEvent` objects for game state tracking.
- **Card lookup**: `CardNameLookup` builds a name→passcode reverse index from `cards.cdb` texts table. Prefers canonical cards (`alias=0`) over alternate artwork variants when multiple rows share the same name.
- **Game state**: `MUDGameState` tracks zone contents (hand, monster, spell/trap, graveyard, banished, extra deck) for both players, plus LP, turn number, and current phase. Updated from `ParsedEvent` objects. Opponent hand tracked as count only (hidden information). Supports periodic resync from `score`/`h`/`tab`/`tab2`/`grave`/`grave2`/`removed`/`removed2`/`extra`/`extra2` command responses to detect and correct tracking drift.
- **Atomic handlers**: `IdleCmdHandler` and `BattleCmdHandler` implement probe→build→decide→execute: they probe the MUD server for available actions, build a `StructuredAction` list (matching the RL flat action space), call `agent.choose()` exactly once, then execute the multi-step MUD conversation. `StructuredAction` encodes `(category, cardspec, card_code, location, sequence, sub_action)` — the same `(card, action_type)` pairs the RL model was trained on.
- **Agent + translator**: Two-layer design separating strategy from protocol. `Agent.choose(prompt, game_state=None) → int` decides *what* to do; `ActionTranslator.translate(action, prompt) → str` converts to MUD text. `PassiveAgent` always ends phases; `RandomAgent` picks uniformly from `structured_actions`. For IDLE_CMD/BATTLE_MENU prompts, agents use `prompt.structured_actions` (populated by handlers). The translator is bypassed for idle/battle — handlers send MUD commands directly.
- **Duel-end detection**: `is_duel_end(line)` matches "You won", "You lost", "You scooped", "was cancelled" patterns to transition from DUEL to FINISHED state.
- **Known limitation — multi-effect cards**: Cards with multiple activatable effects (va/vb) get a single `StructuredAction` with `sub_action="v"`; the handler always picks the first effect. Future model agents needing per-effect choice will require one `StructuredAction` per effect.

## Environment Variables

- `YUGIOH_LIB_PATH` — path to `libocgcore.dylib/.so` (auto-detected from `build/` if unset)
- `YUGIOH_DB_PATH` — path to `cards.cdb` (default: `assets/cards.cdb`)
- `YUGIOH_OPPONENT` — opponent spec: `random`, `greedy`, or `model:path/to/checkpoint.pt` (default: `random`)
- `YUGIOH_OPPONENT_DEVICE` — device for model opponent inference: `cpu` or `cuda` (default: `cpu`)
