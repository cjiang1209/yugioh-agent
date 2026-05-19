# Yu-Gi-Oh! Agent

A Yu-Gi-Oh! research playground built on the ygopro-core engine. Train RL
agents against the real ruleset, play them through a FastAPI server, score
checkpoints on a versioned leaderboard, drive a MUD bot, or run the whole
thing through a web UI.

Powered by the edo9300 fork of ygopro-core, wrapped in Python via ctypes
and exposed as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
environment.

## Quick start

```bash
git clone --recursive https://github.com/cjiang1209/yugioh-agent.git
cd yugioh-agent
./setup.sh                          # venv + deps + libocgcore + cards.cdb + strings.conf

# Start the server, then play a duel interactively
scripts/start_server.sh &
scripts/play_client.sh
```

Requires Python 3.10+, `clang++` with C++17, and `sqlite3` headers.
See [Prerequisites](#prerequisites) for the full list.

## What can I do with this?

### Train an RL agent
PPO trainer that calls the env in-process across N worker subprocesses.
Reward shaping, multi-deck pools, semantic card embeddings, and
snapshot-pool self-play are all CLI flags. Each run gets its own
timestamped directory under `--base-dir`.

```bash
pip install -e ".[train]"
scripts/train.sh --num-envs 8 --total-timesteps 1000000
```

#### Evaluate a checkpoint
Run the same evaluation primitive the trainer uses, against any mix of
agent and opponent specs. Useful for sanity-checking a new heuristic or
picking the better of two models.

```bash
scripts/eval.sh --agent model:checkpoints/run/latest.pt \
    --opponents random greedy --episodes 100
```

#### Score and compare on the leaderboard
Add a checkpoint to score it against a versioned panel of opponents,
then run paired-bootstrap comparisons to answer "does feature X help?"
Group entries by any config field or by user-supplied tags.

```bash
scripts/leaderboard.sh add checkpoints/<run>/checkpoint_latest.pt
scripts/leaderboard.sh compare --by rnn_type
```

### Play a duel from the terminal
Start the FastAPI server and connect with the play client. The server's
opponent can be a random or greedy heuristic, **or one of your trained
checkpoints** via `--opponent model:PATH`.

```bash
# Default: play interactively against a random opponent
scripts/start_server.sh &
scripts/play_client.sh

# Play against your trained model
scripts/start_server.sh --opponent model:checkpoints/run/latest.pt &
scripts/play_client.sh
```

### Play in a browser
React + tRPC web UI for dueling through the same env over HTTP.
Requires `node` + `pnpm` (install via [nvm](https://github.com/nvm-sh/nvm)
and `npm install -g pnpm`).

```bash
# 1. Build the web bundle (installs JS deps + compiles)
scripts/build_web.sh

# 2. Start the Python env server (same as the play client uses)
scripts/start_server.sh &

# 3. Start the web server, then open http://localhost:5000
scripts/start_web.sh
```

### Run a bot in a multiplayer MUD
Connect two bots to a [yugioh-game](https://github.com/tspivey/yugioh-game)
MUD server (a third-party Twisted-based text MUD) that we build locally
into `third_party/`. The bots play complete duels through text protocol —
login, RPS, go-first, and the duel itself.

```bash
scripts/build_mud_server.sh && scripts/start_mud_server.sh &
scripts/seed_mud_accounts.sh
scripts/mud_bot.sh --profile host &
scripts/mud_bot.sh --profile guest
```

## Architecture

```
                       ┌─────────────────────┐
                       │  third_party/       │
                       │  ygopro-core (C++)  │
                       └──────────┬──────────┘
                                  │ ctypes
                       ┌──────────▼──────────┐
                       │  yugioh_core        │  shared primitives:
                       │  + yugioh_env       │  duel state, observations,
                       └──────────┬──────────┘  action mapping
                                  │
       ┌──────────────┬───────────┼────────────┬──────────────┐
       │              │           │            │              │
┌──────▼─────┐ ┌──────▼─────┐ ┌───▼────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ FastAPI    │ │ yugioh_rl  │ │yugioh_ │ │ yugioh_mud │ │ yugioh_web │
│ server     │ │ (PPO)      │ │leader- │ │ (MUD bot)  │ │ (React +   │
│ + clients  │ │            │ │board   │ │            │ │  tRPC)     │
└────────────┘ └────────────┘ └────────┘ └────────────┘ └────────────┘
                                              │
                                   ┌──────────▼──────────┐
                                   │ third_party/        │
                                   │ yugioh-game         │
                                   │ (separate ygopro    │
                                   │  fork, MUD server)  │
                                   └─────────────────────┘
```

`yugioh_core` and `yugioh_env` are the foundation: the C++ engine wrapped
in Python with an observation/action space suitable for RL. Everything
else builds on top — training and leaderboard call the env directly
in-process, the FastAPI server exposes it over HTTP for the play client
and web UI, and the MUD bot is the one outlier (it talks to a separate,
third-party MUD server that uses a different ygopro fork).

| Module                | What it is                                                   |
| --------------------- | ------------------------------------------------------------ |
| `yugioh_core`         | Shared primitives: card DB, observation encoding, constants  |
| `yugioh_env`          | Duel wrapper around ygopro-core + FastAPI server             |
| `yugioh_rl`           | PPO trainer, network, vec-env, eval primitives               |
| `yugioh_leaderboard`  | Versioned panel scoring + paired-bootstrap comparisons       |
| `yugioh_mud`          | Async WebSocket bot for the third-party MUD server           |
| `yugioh_web`          | React + tRPC web UI, HTTP bridge to the FastAPI env          |
| `cli/`                | Argparse entry points behind the `scripts/*.sh` wrappers     |
| `third_party/`        | ygopro-core (submodule), CardScripts (submodule), yugioh-game (cloned on demand) |

## Prerequisites

- **Python** 3.10+
- **C++ toolchain** — `clang++` with C++17 support
- **SQLite** development headers (for compiling the engine wrapper)
- **Git submodules** — pulled by `setup.sh`:
  - [edo9300/ygopro-core](https://github.com/edo9300/ygopro-core) — the duel engine
  - [ProjectIgnis/CardScripts](https://github.com/ProjectIgnis/CardScripts) — Lua card scripts
- **Card database** — `assets/cards.cdb` (downloaded by `setup.sh` on first
  run, or grab it manually from
  [ProjectIgnis/BabelCDB](https://github.com/ProjectIgnis/BabelCDB))
- **String labels** — `assets/strings.conf` (downloaded by `setup.sh` from
  [ProjectIgnis/Distribution](https://github.com/ProjectIgnis/Distribution);
  without it, effect labels fall back to placeholders)

Optional, depending on which subsystems you use:

- **Training** — `pip install -e ".[train]"` (PyTorch + TensorBoard)
- **Semantic card embeddings** — `pip install -e ".[embed]"`
  (sentence-transformers)
- **MUD bot + server** — `pip install -e ".[mud]"` (websockets) plus
  the [tspivey/yugioh-game](https://github.com/tspivey/yugioh-game) MUD
  server, built locally by `scripts/build_mud_server.sh` into
  `third_party/yugioh-game/` (installs into a separate venv; Python 3.12+
  patches applied automatically)
- **Web UI** — `node` + `pnpm` (install via
  [nvm](https://github.com/nvm-sh/nvm) and `npm install -g pnpm`)

## Testing

```bash
make test                                    # full suite
python -m pytest tests/rl/ -v                # one module's tests
python -m pytest tests/env/test_duel.py -v   # one file
```

Tests are organized by module under `tests/` (`core/`, `env/`, `rl/`,
`mud/`, `cli/`, `leaderboard/`). Tests auto-skip when their prerequisites
are missing — e.g., RL tests skip without `torch` installed, engine tests
skip without `libocgcore` built or `cards.cdb` present. Pure unit tests
(parsers, encoders, action space) run with no external dependencies.
