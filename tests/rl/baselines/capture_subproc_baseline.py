"""Capture a frozen subproc-rollout baseline for the bit-equality regression.

Run from the project root:

    python tests/rl/baselines/capture_subproc_baseline.py

Produces ``tests/rl/baselines/subproc_rollout_baseline.npz``, read by
``tests/rl/test_subproc_vec_env.py::test_training_rollout_numerics_unchanged``.

Regenerate the fixture **only** when the env-wrapper / vec-env wire has a
deliberately observable behavior change that you want to lock in as the
new baseline.  After regeneration:

    python -m pytest tests/rl/test_subproc_vec_env.py -v

must pass on the same commit, and the diff to the .npz file should be
reviewed and committed alongside the code change.

The capture loop mirrors the test's loop exactly (``step()`` then
``reset_done()``) so the captured obs hashes correspond to post-reset-done
obs at every index.  The window must cross ≥3 ``done=True`` events so the
comparison exercises the substitution path; the script asserts this
before writing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# tests/rl/baselines/<this_file> → project root is three parents up.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.rl.conftest import hash_obs_field
from yugioh_rl.env_wrapper import SubprocVecEnv, parse_deck_pool

# ----- fixture configuration (pin all knobs so the baseline is reproducible) -----

NUM_ENVS = 4
SEED = 42
DECK_PATH = "assets/decks/starter.ydk"
OPPONENT = "random"
REWARD_SHAPING = False
MIN_DONE_EVENTS = 3
# Cap so a wedged config can't loop forever; chosen large enough that
# random-vs-first-legal episodes (~50-200 steps each) usually hit 3 dones.
MAX_STEPS = 800
OUTPUT = ROOT / "tests" / "rl" / "baselines" / "subproc_rollout_baseline.npz"


def main() -> None:
    deck_path = ROOT / DECK_PATH
    if not deck_path.exists():
        print(f"missing {deck_path}", file=sys.stderr)
        sys.exit(1)
    deck_pool = parse_deck_pool([str(deck_path)])

    vec_env = SubprocVecEnv(
        num_envs=NUM_ENVS,
        deck_pool=deck_pool,
        opponent=OPPONENT,
        reward_shaping=REWARD_SHAPING,
        seed=SEED,
        agent_player="first",   # deterministic — no coin-flip variance
    )

    try:
        obs = vec_env.reset()

        # Per-step capture
        actions_log: list[np.ndarray] = []
        rewards_log: list[np.ndarray] = []
        dones_log: list[np.ndarray] = []
        obs_hashes_log: dict[str, list[list[str]]] = {k: [] for k in obs}

        done_events = 0
        step = 0
        while done_events < MIN_DONE_EVENTS and step < MAX_STEPS:
            # Deterministic policy: first legal action per env (np.argmax of
            # int8 mask returns first 1 if any, else 0). Same selection on
            # baseline-capture and post-refactor replay.
            actions = np.argmax(obs["action_mask"], axis=1).astype(np.int64)
            next_obs, rewards, dones, _infos = vec_env.step(actions)
            # Mirror the test's flow exactly so the captured hashes match
            # what the test will produce.
            next_obs = vec_env.reset_done(dones, next_obs)

            actions_log.append(actions.copy())
            rewards_log.append(rewards.copy())
            dones_log.append(dones.copy())
            for k in obs:
                obs_hashes_log[k].append([
                    hash_obs_field(next_obs[k][i]) for i in range(NUM_ENVS)
                ])

            done_events += int(dones.sum())
            obs = next_obs
            step += 1

        if done_events < MIN_DONE_EVENTS:
            print(
                f"FAILED to observe {MIN_DONE_EVENTS} done events in "
                f"{MAX_STEPS} steps — captured {done_events}. Tune the "
                f"opponent/policy or raise MAX_STEPS.",
                file=sys.stderr,
            )
            sys.exit(2)

        actions_arr = np.stack(actions_log)             # (T, N)
        rewards_arr = np.stack(rewards_log)             # (T, N)
        dones_arr = np.stack(dones_log)                 # (T, N)
        obs_hashes_arrs = {
            f"obs_hashes_{k}": np.array(obs_hashes_log[k], dtype="U40")
            for k in obs_hashes_log
        }

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            OUTPUT,
            num_envs=np.array(NUM_ENVS),
            seed=np.array(SEED),
            actions=actions_arr,
            rewards=rewards_arr,
            dones=dones_arr,
            **obs_hashes_arrs,
        )
        print(
            f"captured {step} steps, {done_events} done events across "
            f"{NUM_ENVS} envs → {OUTPUT}",
        )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
