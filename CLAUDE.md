# CLAUDE.md

Agent-facing guidelines, conventions, and gotchas for this repository.
For human-facing project overview, install instructions, and per-subsystem
usage, see `README.md`.

## Critical Implementation Details

1. **ctypes GC safety**: `DuelCallbacks` stores all ctypes arrays (setcode arrays, script content) as instance attributes to prevent Python GC from freeing memory still referenced by the C engine. Do not remove these storage dicts.

2. **Lua compiled as C++**: The embedded Lua sources are compiled with `clang++ -x c++` to match ygopro-core's C++ symbol linkage. This must not be changed to plain C compilation.

3. **Message framing**: The edo9300 fork prefixes each message buffer entry with a 4-byte `uint32 LE` length field, differing from original ygopro protocol.

4. **Thread safety**: `YuGiOhEnvironment.SUPPORTS_CONCURRENT_SESSIONS = False`. One instance must not run concurrent sessions.

5. **Deck shuffling**: The engine's `Startup` processor clears shuffle flags before the opening draw, so `Duel._add_deck_cards()` shuffles the main deck in Python (using a seeded `random.Random`) before inserting cards. Extra deck cards must be listed under `#extra` in `.ydk` files; cards in the `main` list are always added to the main deck.

6. **Seed handling**: Seeds are spread across 4 `uint64` slots (xoshiro256** RNG) via LCG mixing. Zero seeds are mapped to 1 (engine requires non-zero).

7. **Terminal observations carry the real final board**: `_make_terminal_observation` (`yugioh_env/server/yugioh_environment.py`) passes `query_fn` into `build_observation` just as `_make_observation` does, so with a live duel `cards` holds the end-of-duel board; only `actions`/`action_mask` are zeroed, since a terminal step has no active prompt. Everything zeroes when there is no duel at all. Detect terminal from `done`, never from an empty or all-zero observation field.

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

## Subsystem Gotchas

### RL Training (`yugioh_rl/`)

- **In-process environment**: `TrainingEnv` wraps `YuGiOhEnvironment` directly, bypassing HTTP serialization overhead. Observations are kept as numpy arrays.
- **Subprocess vectorization**: `SubprocVecEnv` spawns N worker processes (one `TrainingEnv` each) using `multiprocessing.spawn` context, respecting the single-session constraint.
- **Shared card embedding**: The same `nn.Embedding` encodes board card IDs and action card codes so the network learns a single card representation.
- **Auto-reset removed**: `TrainingEnv.step()` does **not** auto-reset on episode end; callers must explicitly `reset()` when `done=True`.
- **Player order randomization**: With `--agent-player random` (default), the agent randomly goes first or second per episode. The observation/network is already player-agnostic (relativized by `agent_player`).
- **Semantic card embeddings**: `--card-embeddings` switches the network to a frozen text-embedding lookup + collision-free learned embedding. The embeddings file lives only in the trainer process — `SubprocVecEnv` workers never load it. Symbolic and semantic checkpoints are not interchangeable; `PPOTrainer._validate_checkpoint_compat` enforces this.
- **Resume episode-seed divergence**: On `--resume`, `SubprocVecEnv` is created with the original `config.seed` and `vec_env.reset()` replays the episode seed sequence from the beginning, not from where the interrupted run left off. Training is unaffected (the model still learns), but the exact episode ordering will differ from a single uninterrupted run. Saving and restoring per-env RNG state is impractical given the multi-process architecture.
- **Multi-deck training**: `--deck-paths` accepts multiple `.ydk` files. Each episode, agent and opponent decks are sampled independently from the pool using a per-worker `random.Random(seed)` RNG (separate from the duel RNG). `TrainingEnv.reset()` pre-resolves the `agent_player` coin flip before assigning decks to engine player 0/1, so per-deck metrics are correctly attributed to the agent's deck regardless of turn order.
- **Snapshot-pool self-play**: `--self-play` enables a FIFO ring buffer of past agent snapshots as opponents (default size 10 via `--self-play-pool-size`, snapshots taken at `--save-interval` boundaries). Snapshot opponents use `NetworkOpponent(stochastic=True, temperature=--self-play-temperature)` — softmax sampling rather than greedy argmax. Cross-process sharing via `SharedPolicyWeights` (shared tensors + monotonic version-counter seqlock): trainer calls `publish(net)`; workers `refresh_into(local_net)` lazily on `pool.sample()` and retry on mid-read version bumps. On `--resume`, the pool is reconstructed by replaying interval-aligned numbered checkpoints (`OpponentPool.from_resume`); off-interval crash saves are skipped.
- **Self-play Elo (logging only)**: `OpponentPool` tracks a per-pool Elo rating for the trainer and per-slot ratings (`yugioh_rl/elo.py`, default K=16). On episode end, `TrainingEnv.step()` calls `pool.report_result(slot, agent_won=reward>0)`. Ratings live in shared memory and `PPOTrainer` reads them at log time to emit `selfplay/elo_*` scalars. Updates are intentionally non-atomic across workers (no seqlock): two workers ending episodes simultaneously can race on `agent_rating` and `n_games`, but per-update drift is bounded by K and self-corrects on the next match. Do **not** gate any correctness-sensitive behavior (e.g. snapshotting cadence) on `n_games`. **Resume discontinuity**: `OpponentPool.from_resume` does not restore Elo state — `agent_rating` resets to 1500 on each resume, so the TensorBoard curve jumps at resume boundaries. Same trade-off as the episode-seed divergence above: training is unaffected, only the logged metric.
- **Self-play opponent sampling**: `--self-play-sampling` chooses how the trainer picks an opponent from the snapshot pool. `uniform` (default) preserves prior behavior — every occupied slot equally likely. `pfsp` enables Prioritized Fictitious Self-Play: each slot is weighted by `(1 - P(agent beats slot))^2` using the live Elo ratings, with a uniform-exploration mix and an automatic uniform fallback when the agent dominates every slot. Constants `_PFSP_P` and `_PFSP_EPSILON` at the top of `yugioh_rl/opponent_pool.py` are not exposed as CLI flags — tune by editing them. PFSP reads `agent_rating` and `ratings[]` non-atomically (same trade-off as the Elo update path); drift is bounded by K per update and self-corrects, so sampling pressure is noisy but unbiased.
- **Async actor-learner**: `--vec-env-type async_actor_learner` runs workers continuously without sync barriers. Workers push completed rollouts to a `multiprocessing.Queue`; the trainer drains K qualifying rollouts per update. `--max-version-lag N` (default 5) discards rollouts more than N updates behind the trainer's current weights. V-trace (Espeholt et al., 2018) replaces GAE for advantage estimation, correcting for off-policy data via truncated importance sampling (`--vtrace-rho-bar`, `--vtrace-c-bar`, both default 1.0). Workers refresh weights from `SharedPolicyWeights` at rollout boundaries only when the version has changed; otherwise they carry hx forward. Monitor `async/version_lag_mean`, `async/rollouts_discarded`, and `async/queue_depth` in TensorBoard.
- **Async vs sync actor-learner**: Both use `SharedPolicyWeights` for weight transport. Sync (`sync_actor_learner`) has a barrier: trainer sends `("go", v)`, waits for all N rollouts, then updates. Async (`async_actor_learner`) eliminates the barrier: fast workers start their next rollout immediately instead of idling while the slowest worker finishes (wall-clock time per rollout varies with game-state complexity even though step count is fixed).
- **`--config` JSON loading**: `cli/train.py --config PATH` loads partial `TrainingConfig` fields from JSON; CLI flags always override JSON. `save_dir` and `resume_checkpoint` are dropped on load (derived from `--base-dir` and `--resume`). `--config` is mutex with `--resume` (resume loads config from the checkpoint).
- **Eval determinism**: `--workers N` for `cli/eval.py` is deterministic — workers fan out across `(opponent, episode_idx)` tuples; results aggregated sorted by `episode_idx` per opponent so `--workers 1`, `2`, `4` produce byte-equal `EvalResult`s on the same checkpoint.
- **Pending chain buffer**: `--chain-embed-dim N` (default 32) adds a `(8, 16)` uint8 buffer to the observation that shows the current pending chain links during chain building. Each entry encodes the card code, effect descriptor, controller, location, sequence, and chain link number. When the agent is asked to chain (MSG_SELECT_CHAIN), it can see exactly what it's chaining onto. The buffer is maintained in `GameState`, cleared on MSG_CHAIN_END, and encoded through the standard observation pipeline. The network embeds chain entries using the existing card embedding table and a small encoder MLP, mean-pools across entries, and concatenates with the board representation. Set to 0 to disable.

### Leaderboard (`yugioh_leaderboard/`)

- **Single user, single process.** No file locks. Don't run two `add` commands in parallel — entries are safe (unique filenames) but the index regen is last-writer-wins.
- **Entry deletion is `rm`.** No `delete` subcommand — by design (per-file storage chosen specifically to make `rm` natural).
- **Panel changes are user-driven.** Edit `leaderboard.config.json` by hand to bump `panel_version` and move retired panels into `history`. Old entries get flagged "stale" in `index.md` and excluded from `compare` (use `--include-stale` to override).
- **Pairwise re-runs are commutative.** `pairwise A B` and `pairwise B A` produce the same seed and overwrite the prior record (matched by `vs_entry_id`).
- **Pairwise needs deck overlap.** Default decks are the intersection of both entries' deck pools; pass `--decks` to override when entries share none (otherwise `NoSharedDecksError`).
- **Pairwise assumes no ties.** The mirror computes `b_wins = episodes - r.wins`, so ties would silently produce negative B-wins; the engine never returns ties in practice. `run_pairwise` only asserts the weaker `wins <= episodes` invariant as a sanity check.
- **`--workers N` is deterministic.** Both `add` and `pairwise` accept `--workers N` (default 1). Each worker re-loads `model:` checkpoints, so memory scales linearly with N — sizing rule is `N ≤ available_RAM / per_model_size`.

### MUD Server (third-party `tspivey/yugioh-game`)

- **Different ygopro-core fork**: Fluorohydride (C API: `create_duel`, `process`, `set_responsei`) vs the main project's edo9300 (`OCG_CreateDuel`, `OCG_DuelProcess`, `OCG_DuelSetResponse`). Completely incompatible APIs.
- **Shared library**: `libygo.so` (CFFI) vs `libocgcore.dylib/.so` (ctypes). No conflict.
- **Separate venv**: `third_party/yugioh-game/.venv` — built by `scripts/build_mud_server.sh` with deps that conflict with the main project. Upstream's pinned `requirements.txt` (Twisted 18.4.0, SQLAlchemy 1.3.4, etc.) is too old for Python 3.10+; the build script installs unpinned `twisted` + `sqlalchemy>=1.4,<2` + `attrs<24` instead.
- **Cloned on demand**: Not a git submodule. `third_party/yugioh-game/` is gitignored, treated as a build artifact.
- **Lua 5.3.5**: Downloaded and compiled by the build script (compiled as C++ with `CC=clang++` for C++ linkage).
- **Python 3.12+ patches** (applied automatically by `build_mud_server.sh`):
  - `duel.py`: `pkgutil.iter_modules()` returns `FileFinder` objects that lack `find_module()` (removed in 3.12). Patched to use `importlib.import_module()`. Without this, all message handlers fail to load silently and duels hang.
  - `gsb/command.py`: `re._pattern_type` was removed in 3.7. Patched to use `re.Pattern`.
  - `gsb/intercept.py`: `Reader.done` attribute reordering under attrs>=22. When `done` is overridden with a default, it moves to the end of the attribute list, causing positional args to be misrouted. Patched with `__attrs_post_init__` to detect and fix the misrouted callable.

### MUD Bot Client (`yugioh_mud/`)

- **Known limitation — multi-effect cards**: Cards with multiple activatable effects (va/vb) get a single `StructuredAction` with `sub_action="v"`; the handler always picks the first effect. Future model agents needing per-effect choice will require one `StructuredAction` per effect.
- **Known limitation — sort prompts (MSG_SORT_CARD)**: The MUD `Agent`
  protocol returns a single int per prompt, which can't express an N-element
  permutation. All MUD agents submit the identity permutation `[1, 2, ..., N]`
  for sort prompts. A model trained against the in-process engine (which uses
  the multi-step pick harness) cannot meaningfully drive sort decisions
  through the MUD bot. Restructuring would require a new agent-return shape
  for sort, deferred until needed.

### Puzzle State Initialization (`yugioh_env/puzzle.py`)

- **Engine-native disable**: Cards marked `disabled: True` are disabled via a Lua `EFFECT_DISABLE` effect loaded through `OCG_LoadScript`, not by bit manipulation. The engine's `refresh_disable_status()` handles all propagation.
- **`starting_draw=0`**: `create_puzzle()` sets both players' `startingDrawCount` to 0 so the Startup processor doesn't draw cards — all hand contents come from the puzzle spec.
- **Player 0 always goes first**: The engine starts with player 0's turn. Use `agent_player=1` in `env.reset()` to have the agent go second — `_process_to_agent_choice` auto-plays player 0 (opponent) naturally.
- **Deck order**: Deck cards are inserted in specification order (no shuffle). The engine stores decks bottom-to-top internally, so the first card in the spec's `deck` list is drawn first.
- **`disabled` on field only**: The `disabled` flag is only valid on `monster_zone` and `spell_zone` entries. The schema validator rejects it on non-field zones.

### ygo-agent Bridge (`yugioh_env/ygo_agent/`)

- **`selected` is positions into the list sent; `response` is the engine
  index.** They coincide unless sum pruning or `MAX_ACTIONS` truncation drops
  a card.
- **The request is golden-pinned** — regenerate
  `tests/env/fixtures/ygo_agent_predict_requests.json` only via the capture
  script beside it. Prompts with no server schema go in
  `_SERVER_UNSUPPORTED_MSGS`; one in neither that nor the translator table
  raises mid-duel.

### Deterministic Replay (`yugioh_env/replay.py`)

- **Non-intrusive**: Recording and replay wrap `YuGiOhEnvironment` and the opponent without modifying either class.
- **Interleaved action log**: Both players' actions are stored in game order as `(msg_type, player, action, num_actions)` entries. The interleaved order is the source of truth for drift detection.
- **Recording**: `RecordingEnvironment` wraps the env and records agent actions. `RecordingOpponent` wraps the opponent and records opponent actions. Both write to the same `GameRecording`.
- **Replay**: `ScriptedOpponent` reads opponent actions from a `ReplayCursor`. The caller drives agent actions via `env.step()` using the same cursor. Drift detection raises `RuntimeError` if msg_type or player doesn't match.
- **Serialization**: `GameRecording.save()` / `GameRecording.load()` use JSON. The `setup` dict contains either deck paths + seed (for normal games) or a `puzzle` key (for puzzle states).

## Environment Variables

- `YUGIOH_LIB_PATH` — path to `libocgcore.dylib/.so` (auto-detected from `build/` if unset)
- `YUGIOH_DB_PATH` — path to `cards.cdb` (default: `assets/cards.cdb`)
- `YUGIOH_OPPONENT` — opponent spec: `random`, `greedy`, or `model:path/to/checkpoint.pt` (default: `random`)
- `YUGIOH_OPPONENT_DEVICE` — device for model opponent inference: `cpu` or `cuda` (default: `cpu`). Read by the FastAPI server (`scripts/start_server.sh`), the training rollout (`SubprocVecEnv` workers), and in-training eval (`PPOTrainer._evaluate`). **Not** read by the standalone eval CLI — `scripts/eval.sh --device` is the explicit override there.
- `YUGIOH_RECOMMENDER` — action recommender for the web UI's "AI Assist"
  feature. Accepts the full opponent grammar: `random`, `greedy`,
  `model:path/to/checkpoint.pt`, or `ygo-agent[:url]` (unset = feature
  unavailable). Loaded once at FastAPI startup, independent of
  `YUGIOH_OPPONENT`; suggests moves for the *human* player.
- `YUGIOH_RECOMMENDER_DEVICE` — device for `model:` recommender inference:
  `cpu` or `cuda` (default: `cpu`).
- `VITE_API_BASE` — backend base URL baked into the web bundle at build time
  (default `http://localhost:8000`). Only needed when the FastAPI server runs
  somewhere other than the local `scripts/start_server.sh`.
- `MLFLOW_TRACKING_URI` — mlflow tracking server, read by the mlflow sink when
  `--log-to` includes `mlflow`. `scripts/train.sh` and `scripts/eval_sweep.sh`
  default it to `http://127.0.0.1:5000/` (override by exporting before invoking).
- `MLFLOW_EXPERIMENT_NAME` — mlflow-native env var selecting the experiment for
  training/eval runs (default `yugioh`). Export it to group runs under a custom
  experiment.

**Port map** (avoid collisions): mlflow server `5000`, web UI (`scripts/start_web.sh`)
`7000`, FastAPI backend (`scripts/start_server.sh`) `8000`, TensorBoard `6006`,
ygo-agent bridge `3000`, MUD websocket `8080`.

## Test Skip Behavior

Tests are organized into subdirectories by module (`tests/core/`, `tests/env/`, `tests/mud/`, `tests/rl/`, `tests/cli/`, `tests/leaderboard/`). Run a single module's tests with e.g. `python -m pytest tests/mud/ -v`.

The web suite (`yugioh_web/`, 16 vitest files) runs via `make test-web`, which
`make test` includes. It **soft-skips** with a loud banner when node, pnpm or
`node_modules` is missing, so a toolchain-free checkout still reports success;
`STRICT_WEB=1 make test-web` turns that skip into a failure.

Tests auto-skip when prerequisites are missing:
- `lib` fixture (`tests/env/conftest.py`): skips if `libocgcore` not built (`make build`)
- `db_path` fixture (`tests/conftest.py`): skips if `assets/cards.cdb` absent
- `script_dirs` fixture (`tests/env/conftest.py`): skips if `third_party/CardScripts/` absent
- `tests/env/test_opponent.py` ModelOpponent tests: skips if `torch` not installed
- `tests/rl/test_card_embeddings.py`: TextEmbeddingLookup/network tests skip if `torch` not installed; `test_build_embeddings_output_structure` skips if `sentence-transformers` not installed
- `tests/rl/test_checkpoint_init.py`: skips if `torch` not installed
- `tests/rl/test_eval_opponents.py`: skips if `torch` not installed
- `tests/rl/test_multi_deck.py`: skips if `torch` not installed
- `tests/rl/test_resume.py`: skips if `torch` not installed
- `tests/mud/test_model_agent.py`: skips if `torch` not installed
- `tests/leaderboard/test_features.py`: skips if `torch` not installed
- `tests/leaderboard/test_pairwise.py::test_pairwise_mirrors_symmetrically`: skips if `libocgcore` or `assets/cards.cdb` missing (other tests in the file run unconditionally)
- `tests/leaderboard/test_score.py`: skips if `libocgcore` or `assets/cards.cdb` missing

All other test files run unconditionally (pure unit tests with no external deps).
