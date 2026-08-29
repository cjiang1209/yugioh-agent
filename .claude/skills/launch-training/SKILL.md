---
name: launch-training
description: Use when starting a training run — user says "train", "start training", "launch training", "run training", or passes a --config / config.json for cli/train.py. Runs pre-flight checks before executing.
---

# Launch Training

Guide the user through training mode selection, parameter review, and pre-flight validation before launching.

## Step 1 — Training Mode

Three modes with different parameter flows:

### 1a. Ask training mode

Ask this first, before touching the filesystem. Use `AskUserQuestion` with options:
- `Start fresh` — new run from scratch, all parameters asked
- `Init from checkpoint` — new run with weights from an existing checkpoint, all parameters asked (architecture must match)
- `Resume` — continue an interrupted run, most parameters locked to the original config

If the user's original message already specifies `--resume` or a checkpoint to resume, skip to the appropriate mode.

### 1b. Find recent checkpoints (init / resume modes only)

Scan `$PROJECT_DIR/checkpoints/` for subdirectories containing
`checkpoint_latest.pt` or `config*.json`, sorted by modification time. Read the
config JSON of the two most recent only — Step 2 offers those two plus Other,
and an Other pick can be read when the user names it.

## Step 2 — Parameter Review

The questions asked depend on the mode chosen.

### Mode: Start fresh

Ask all parameters in two submissions.

### Mode: Init from checkpoint

Ask user to select a checkpoint (show the two most recent, plus Other). Then ask all parameters in two submissions — architecture params (embedding mode, RNN) must match the checkpoint.

### Mode: Resume

Ask user to select a checkpoint to resume (show the two most recent with `checkpoint_latest.pt`, plus Other). Only these parameters can be overridden on resume — ask them in a single submission:
- **Total timesteps** (to extend training)
- **Device** (to switch hardware)
- **Learning rate** (to adjust)

All other parameters are locked to the checkpoint's saved config. Display the locked config for reference before asking overrides.

---

### Submission 1: Core parameters (fresh / init modes only)

Ask together in one `AskUserQuestion` call (up to 4 questions, skip covered ones):

**Decks** (skip if config provides `deck_paths`)

Scan `$PROJECT_DIR/assets/decks/*.ydk` and list all available decks in a markdown table before asking. Options:
- `All decks` — use every `.ydk` in the directory (Recommended for multi-deck training)
- `Custom selection` — ask user to list deck names

**Device** (skip if config provides `device`)

Options:
- `auto` — CUDA if available, else MPS, else CPU (Recommended)
- `mps` — Apple Silicon GPU
- `cpu` — CPU only

**Total timesteps** (ALWAYS ask, even if config provides it)

Options:
- `1M` — quick test (~20 min)
- `5M` — standard run (~2 hours) (Recommended)
- `10M` — long run (~4 hours)
- (Other for custom)

If the loaded config has a `total_timesteps` value, show it in the description of the matching option or add it as an explicit option if it doesn't match any preset.

**Vec-env type and number of environments** (skip if config provides both `vec_env_type` and `num_envs`)

One question — the environment count only makes sense against the transport.

Options:
- `async_actor_learner · 8 envs` — best per-env CPU use, no barrier (Recommended)
- `sync_actor_learner · 16 envs` — workers hold a local policy, trainer waits for all N before updating
- `subproc · 32 envs` — trainer runs inference centrally with per-step IPC; simpler but slower, and crashes with self-play
- (Other for any combination the user prefers)

If the config supplies only one of the two, treat the supplied one as fixed and
ask only for its partner.

### Submission 2: Training strategy (fresh / init modes only)

Ask together in one `AskUserQuestion` call (up to 4 questions, skip covered ones):

**Initial opponent** (skip if config provides `opponent`)

Options:
- `greedy` (Recommended)
- `random`
- (Other for `model:<path>`)

**Embedding mode** (skip if config provides `card_embeddings`)

Options:
- `Semantic` — use text embeddings at `assets/card_text_embeddings.pt` (Recommended)
- `Symbolic` — learned embeddings only

**Self-play** (skip if config provides `self_play`)

Options:
- `Yes, with PFSP sampling` (Recommended)
- `Yes, with uniform sampling`
- `No self-play`

**Recommended configuration** — ask last, as a single `multiSelect` question
listing every recommended setting at once. Drop any entry the config already
provides. Pass the Unselected column literally for anything the user leaves
unticked; do not read a blank selection as "use defaults".

| Selected | Unselected |
|---|---|
| `Chain embedding (Recommended)` — the agent sees the chain it is building → `--chain-embed-dim 32` | `--chain-embed-dim 0` |
| `Event-history embedding (Recommended)` — rolling both-players event history feeds the value head → `--event-history-dim 64` | `--event-history-dim 0` |
| `Disable in-training eval (Recommended)` — eval steals rollout throughput → `--eval-interval 999999` | leave `--eval-interval` at its default of 50 |
| `MLflow logging alongside TensorBoard (Recommended)` → `--log-to tensorboard mlflow` | `--log-to tensorboard` |

## Step 3 — Confirmation Summary

Display a summary table of the effective config. For resume mode, mark locked parameters with `(locked)`:

```
┌─────────────────────┬─────────────────────────────────┐
│ Parameter           │ Value                           │
├─────────────────────┼─────────────────────────────────┤
│ Mode                │ fresh / init / resume           │
│ Checkpoint          │ <path> (init/resume only)       │
│ Decks               │ utopia.ydk, blue_eyes.ydk       │
│ Total timesteps     │ 10,000,000                      │
│ Num envs            │ 8                               │
│ Embedding mode      │ Semantic                        │
│ Chain embed dim     │ 32                              │
│ Event history dim   │ 64                              │
│ Self-play           │ PFSP                            │
│ Opponent            │ greedy                          │
│ Eval interval       │ Disabled                        │
│ Logging             │ tensorboard, mlflow             │
│ Device              │ mps                             │
│ Vec-env type        │ async_actor_learner             │
│ Max version lag     │ 5 (async only)                  │
│ V-trace rho_bar     │ 1.0 (async only)                │
│ V-trace c_bar       │ 1.0 (async only)                │
│ RNN                 │ none                            │
│ Seed                │ 42                              │
└─────────────────────┴─────────────────────────────────┘
```

Ask the user to confirm or go back and change parameters:
- `Launch` (Recommended)
- `Change parameters` — go back to Step 2

## Step 4 — Pre-flight Validation

Run ALL checks programmatically. Abort on any failure.

### 4a. Resolve PROJECT_DIR and Python

Determine `PROJECT_DIR` — the repo root containing `cli/train.py`. If config path or cwd is inside a worktree, use that worktree.

**Always use** `$PROJECT_DIR/scripts/train.sh` to launch — never call `python -m cli.train` directly (venv may not be on PATH).

### 4b. Verify build artifacts

Every path 4b-4d needs is known once Step 3's table exists, so test them in one
pass and print only what is missing:

```bash
for p in "$PROJECT_DIR"/build/libocgcore.* "$PROJECT_DIR/assets/cards.cdb" \
         "$PROJECT_DIR/third_party/CardScripts" <each deck> <embeddings .pt> <checkpoint .pt>; do
  [ -e "$p" ] || echo "MISSING $p"
done
```

`libocgcore` missing means `make build`. 4e, 4f, 4g and 4i below are reasoning
over values already in hand — no further calls.

### 4c. Validate deck files

Each deck path must exist on disk and end with `.ydk`.

### 4d. Validate card embeddings

If semantic mode: verify the `.pt` file exists. Warn about checkpoint incompatibility with symbolic mode.

### 4e. Detect known crash combos

| Combo | Result | Fix |
|---|---|---|
| `rnn_type=lstm` + `device=mps` | Hard crash on backward pass | Use `--rnn-type gru` or `--device cpu` |
| `self_play=true` + `vec_env_type=subproc` | Worker crash: `RuntimeError: Connection refused` on shared tensor rebuild | Use `--vec-env-type sync_actor_learner` |

### 4f. Validate TBPTT constraints (if recurrent)

When `rnn_type != "none"`, ALL must hold:
- `rollout_steps % bptt_chunk_len == 0`
- `minibatch_size >= rollout_steps`
- `minibatch_size % rollout_steps == 0`
- `num_envs * rollout_steps >= minibatch_size`

### 4g. Validate resume constraints (if resume mode)

- `--resume` is mutex with `--config` and `--init-checkpoint`
- Checkpoint file must exist and contain `checkpoint_latest.pt`
- Only allowlisted flags can be overridden: `total_timesteps`, `learning_rate`, `device`, `log_interval`, `eval_interval`, `eval_episodes`, `eval_opponents`, `save_interval`, `opponent`

### 4h. Validate init-checkpoint constraints (if init mode)

- Checkpoint `.pt` file must exist
- Architecture fields must match: `card_embed_dim`, `global_embed_dim`, `board_hidden_dim`, `action_embed_dim`, `rnn_type`
- Cannot add text embeddings to a symbolic checkpoint or vice versa

### 4i. Validate opponent specs

Each spec (`--opponent`, `--eval-opponents`) must be `random`, `greedy`, or `model:<existing_path>`.

### 4j. Validate MLflow availability (only if `--log-to` includes mlflow)

```bash
# find_spec, not import — presence is the question, and importing mlflow is slow
$PROJECT_DIR/.venv/bin/python -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('mlflow') is None)" \
  || echo "MISSING mlflow — pip install -e '.[train]'"
URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000/}"    # same default scripts/train.sh exports
curl -fsS --max-time 5 "${URI%/}/health"                # MUST print OK
```

If the server is unreachable, do NOT launch. Report the URI probed and offer:
start the server (`mlflow server --host 127.0.0.1 --port 5000`), point
`MLFLOW_TRACKING_URI` elsewhere, or drop back to `--log-to tensorboard`.

## Step 5 — Launch

Build the CLI command from the effective config and mode:

- **Fresh**: `scripts/train.sh [all flags]`
- **Init**: `scripts/train.sh --init-checkpoint <path> [all flags]`
- **Resume**: `scripts/train.sh --resume <path> [override flags only]`

Launch **detached** — never `run_in_background: true`. Anchor every path on
`$PROJECT_DIR`: the launch is backgrounded, so a wrong path fails silently and
the call still exits 0.

```bash
LOG="$PROJECT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_DIR/logs"

# perl, not `setsid` — stock macOS ships no setsid binary
perl -e 'use POSIX qw(setsid); exit if fork; setsid(); exec @ARGV' \
  "$PROJECT_DIR/scripts/train.sh" [flags] >"$LOG" 2>&1 </dev/null &
```

Then poll for startup and resolve the process group, in one call:

```bash
for _ in $(seq 30); do grep -q 'Starting training:' "$LOG" && break; sleep 1; done
grep -m1 -E 'Run directory:|Starting training:' "$LOG"
ps -eo ppid,pgid,command | awk '$1==1 && /cli\.train/ {print $2}'   # expect one line
```

Two lines from that last command means an earlier run is still alive — resolve
that before continuing.

Report the **PGID**, the **run directory**, the **tail command**, and where to
watch progress: TensorBoard, plus the MLflow URI when 4j ran. The PGID has to
reach the transcript — shell variables do not survive between calls, and
stopping the run needs it. Detached means no completion notification, so say
so explicitly.

## Note — Stopping a run

Applies only when the user asks to stop a run — not part of the launch flow.

Workers exec under the `spawn` context, so **the module name `cli.train`
appears only in the parent's command line.** Killing the parent alone strands
them at ~100% CPU each, reparented to PPID 1 and reporting to nothing. Kill the
group instead, guarding the number first — a stray value signals the wrong
group, up to and including every process the user owns:

```bash
TRAIN_PGID=<the PGID reported at launch>   # else re-derive with the ps|awk from Step 5
[[ "$TRAIN_PGID" =~ ^[0-9]+$ ]] && (( TRAIN_PGID > 1 )) || { echo "refusing: bad PGID"; exit 1; }
kill -- "-$TRAIN_PGID"
for _ in 1 2 3 4 5; do kill -0 -- "-$TRAIN_PGID" 2>/dev/null || break; sleep 1; done
kill -9 -- "-$TRAIN_PGID" 2>/dev/null || true            # only if any survived
ps -eo pid,ppid,%cpu,command | awk '$2==1 && /multiprocessing\.spawn/'
```

Any row from that last command is an orphaned worker — report its CPU, then
kill by PID. Match on `multiprocessing.spawn`, not on the venv path: other
long-lived services run from the same interpreter at PPID 1, and the web
backend would otherwise show up as an orphan. A live run's workers have a live
parent, so `$2==1` already excludes them.

SIGTERM discards progress back to the last numbered checkpoint, up to
`--save-interval` updates. Say which checkpoint the run rewinds to.

## Common Mistakes

| Mistake | Impact | Prevention |
|---|---|---|
| `python -m cli.train` directly | `command not found: python` | Always use `scripts/train.sh` |
| `run_in_background: true` | Run killed when the session ends | Launch detached — see Step 5 |
| `pkill -f cli.train` to stop a run | Orphans every worker at full CPU, and "stopped" is then a false claim | See the stopping note |
| `pgrep -f "cli.train"` to find the trainer | Matches the harness's own shell | Filter on `ppid == 1` — see Step 5 |
| `--log-to mlflow` with no server up | `RuntimeError` after launch | Pre-flight 4j probes `/health` |
| Sizing `num_envs` for async as if it were sync | Oversubscribed CPU, no throughput gain | Envs and vec-env type are one choice |
| Missing CardScripts | Cards don't resolve | Check `third_party/CardScripts/` |
| LSTM on MPS | Crash on backward pass | Detect at pre-flight |
| Changing `deck_paths` on resume | Silent metric misattribution | Blocked — resume locks it |
| Large `num_envs` (64+) | OOM risk | Warn about memory |
| Init with mismatched architecture | `ValueError` at load | Pre-flight checks arch fields |
| Semantic ↔ symbolic mismatch on init | `ValueError` at load | Pre-flight blocks cross-mode init |
