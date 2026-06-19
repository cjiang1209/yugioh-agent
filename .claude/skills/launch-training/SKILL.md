---
name: launch-training
description: Use when starting a training run — user says "train", "start training", "launch training", "run training", or passes a --config / config.json for cli/train.py. Runs pre-flight checks before executing.
---

# Launch Training

Guide the user through training mode selection, parameter review, and pre-flight validation before launching.

## Step 1 — Training Mode

Three modes with different parameter flows:

### 1a. Find recent checkpoints

Scan `$PROJECT_DIR/checkpoints/` for subdirectories containing `checkpoint_latest.pt` or `config*.json`, sorted by modification time (newest first). Read each config JSON to extract key params.

### 1b. Ask training mode

Use `AskUserQuestion` with options:
- `Start fresh` — new run from scratch, all parameters asked
- `Init from checkpoint` — new run with weights from an existing checkpoint, all parameters asked (architecture must match)
- `Resume` — continue an interrupted run, most parameters locked to the original config

If the user's original message already specifies `--resume` or a checkpoint to resume, skip to the appropriate mode.

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

**Initial opponent** (skip if config provides `opponent`)

Options:
- `greedy` (Recommended)
- `random`
- (Other for `model:<path>`)

**Total timesteps** (ALWAYS ask, even if config provides it)

Options:
- `1M` — quick test (~20 min)
- `5M` — standard run (~2 hours) (Recommended)
- `10M` — long run (~4 hours)
- (Other for custom)

If the loaded config has a `total_timesteps` value, show it in the description of the matching option or add it as an explicit option if it doesn't match any preset.

**Number of environments** (skip if config provides `num_envs`)

Options:
- `8` — light, low memory
- `32` — standard (Recommended)
- `64` — fast, high memory (~8GB+)

**Device** (skip if config provides `device`)

Options:
- `auto` — CUDA if available, else MPS, else CPU (Recommended)
- `mps` — Apple Silicon GPU
- `cpu` — CPU only

### Submission 2: Training strategy (fresh / init modes only)

Ask together in one `AskUserQuestion` call (up to 4 questions, skip covered ones):

**Embedding mode** (skip if config provides `card_embeddings`)

Options:
- `Semantic` — use text embeddings at `assets/card_text_embeddings.pt` (Recommended)
- `Symbolic` — learned embeddings only

**Vec-env type** (skip if config provides `vec_env_type`)

Options:
- `sync_actor_learner` — workers hold a local policy and submit full rollouts; eliminates per-step IPC (Recommended)
- `subproc` — synchronous IPC per step; simpler but slower and crashes with self-play due to shared tensor issues

**Self-play** (skip if config provides `self_play`)

Options:
- `Yes, with PFSP sampling` (Recommended)
- `Yes, with uniform sampling`
- `No self-play`

**In-training eval** (skip if config provides `eval_interval`)

Options:
- `Disabled` — set eval_interval to 999999 (Recommended)
- `Every 50 updates` — standard
- `Every 100 updates` — less frequent

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
│ Num envs            │ 32                              │
│ Embedding mode      │ Semantic                        │
│ Self-play           │ PFSP                            │
│ Opponent            │ greedy                          │
│ Eval interval       │ Disabled                        │
│ Device              │ mps                             │
│ Vec-env type        │ sync_actor_learner              │
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

```bash
ls $PROJECT_DIR/build/libocgcore.* 2>/dev/null   # MUST exist, else: make build
ls $PROJECT_DIR/assets/cards.cdb 2>/dev/null      # MUST exist
ls $PROJECT_DIR/third_party/CardScripts/ 2>/dev/null  # SHOULD exist
```

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

## Step 5 — Launch

Build the CLI command from the effective config and mode:

- **Fresh**: `scripts/train.sh [all flags]`
- **Init**: `scripts/train.sh --init-checkpoint <path> [all flags]`
- **Resume**: `scripts/train.sh --resume <path> [override flags only]`

Run in background (`run_in_background: true`). Wait ~10s, check output for `Starting training:` line. Print the **run directory**, **tail command**, and note TensorBoard availability.

## Common Mistakes

| Mistake | Impact | Prevention |
|---|---|---|
| `python -m cli.train` directly | `command not found: python` | Always use `scripts/train.sh` |
| Missing CardScripts | Cards don't resolve | Check `third_party/CardScripts/` |
| LSTM on MPS | Crash on backward pass | Detect at pre-flight |
| Changing `deck_paths` on resume | Silent metric misattribution | Blocked — resume locks it |
| Large `num_envs` (64+) | OOM risk | Warn about memory |
| Init with mismatched architecture | `ValueError` at load | Pre-flight checks arch fields |
| Semantic ↔ symbolic mismatch on init | `ValueError` at load | Pre-flight blocks cross-mode init |
