"""Tests for ``SubprocVecEnv`` — drift detection on the env-wrapper +
vec-env IPC path.

The headline test, :func:`test_training_rollout_numerics_unchanged`, is a
general bit-equality regression: it pins the rollout produced by stepping
``SubprocVecEnv`` with a model-free policy against a frozen baseline at
``tests/rl/baselines/subproc_rollout_baseline.npz``.  Any change that
perturbs determinism on this code path — RNG ordering inside ``reset()``,
``episode_seed`` derivation, obs-encoding field order/dtype, IPC-induced
nondeterminism — surfaces as a step-N divergence with a clear error
message.  The fixture window is sized to cross at least one terminal
transition, so the substitution path through ``reset_done`` is exercised
alongside the non-terminal middle of episodes.

Regenerating the fixture
------------------------
Only when the env wrapper has a deliberately observable behavior change:

    python tests/rl/baselines/capture_subproc_baseline.py
    python -m pytest tests/rl/test_subproc_vec_env.py -v

Both steps must pass on the same commit; review and commit the .npz diff
together with the code change.  The capture script mirrors this test's
loop exactly (step → reset_done → record), so by construction the test
matches the freshly-captured fixture on the current code.

When bit-equality is the right tool — and when it isn't
--------------------------------------------------------
Bit-equality is the right invariant **here** because everything stochastic
is already pinned: a model-free policy (``np.argmax(action_mask)``), fixed
seed, deterministic engine, and Python MT19937 for deck/player RNG.  Under
those conditions the only legitimate sources of variation are bugs.

It is **not** the right tool when learned dynamics or hardware-dependent
floating-point ordering are in the loop.  Do not extend bit-equality to:

  * Cross-implementation comparisons (subproc vs actor-learner) —
    floating-point reduction order across processes diverges.  See
    ``tests/rl/test_actor_learner_equivalence.py`` (param L2 within 20%).
  * Training-pipeline changes — torch version, cuDNN/MPS kernel,
    mixed-precision, batch ordering, and worker spawn timing all perturb
    trained weights without being bugs.  Use a panel/distributional
    metric (paired-bootstrap CIs, leaderboard ``compare``) instead.
  * Cross-platform comparisons (macOS vs Linux, MPS vs CPU) — different
    SIMD/BLAS routines yield different floats from the same code.

Bit-equality where the policy is frozen and reductions are single-threaded;
distributional everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.rl.conftest import hash_obs_field, requires_engine
from yugioh_rl.env_wrapper import SubprocVecEnv, parse_deck_pool

BASELINE_PATH = Path(__file__).parent / "baselines" / "subproc_rollout_baseline.npz"


@requires_engine
def test_training_rollout_numerics_unchanged() -> None:
    """Replay the frozen baseline and assert bit-equal (actions, rewards,
    dones, obs hashes) at every step, including the post-``reset_done``
    substituted obs at each done index.

    The substitution check is what catches "terminal obs leaked back as
    the new-episode obs" regressions — without it, a bug that returned
    the terminal obs from ``reset_done`` would pass actions/rewards/dones.
    """
    if not BASELINE_PATH.exists():
        pytest.skip(f"missing baseline: {BASELINE_PATH}")
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")

    baseline = np.load(BASELINE_PATH)
    num_envs = int(baseline["num_envs"])
    seed = int(baseline["seed"])
    expected_actions = baseline["actions"]  # (T, N)
    expected_rewards = baseline["rewards"]
    expected_dones = baseline["dones"]
    obs_keys = [k.replace("obs_hashes_", "") for k in baseline.files if k.startswith("obs_hashes_")]
    expected_hashes = {k: baseline[f"obs_hashes_{k}"] for k in obs_keys}

    T = expected_actions.shape[0]
    assert int(expected_dones.sum()) >= 3, (
        "baseline did not cross ≥3 done events — re-run "
        "scripts/capture_subproc_baseline.py with a wider window"
    )

    deck_pool = parse_deck_pool([str(deck_path)])
    vec_env = SubprocVecEnv(
        num_envs=num_envs,
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=seed,
        agent_player="first",
    )

    try:
        obs = vec_env.reset()
        for t in range(T):
            # Same deterministic policy used by the baseline-capture script.
            actions = np.argmax(obs["action_mask"], axis=1).astype(np.int64)
            assert np.array_equal(actions, expected_actions[t]), (
                f"actions diverged at step {t}: "
                f"got {actions.tolist()}, expected {expected_actions[t].tolist()}"
            )

            next_obs, rewards, dones, _ = vec_env.step(actions)

            # (i) rewards
            assert np.array_equal(rewards, expected_rewards[t]), f"rewards diverged at step {t}"
            # (ii) dones
            assert np.array_equal(dones, expected_dones[t]), (
                f"dones diverged at step {t}: "
                f"got {dones.tolist()}, expected {expected_dones[t].tolist()}"
            )
            # Substitute new-episode obs at done indices — equivalent of the
            # old auto-reset, just explicit.  Without this, (iii) below would
            # be comparing a terminal obs against a new-episode-start obs.
            next_obs = vec_env.reset_done(dones, next_obs)

            # (iii) full obs (including post-reset substitution at done indices)
            for k in obs_keys:
                for i in range(num_envs):
                    got = hash_obs_field(next_obs[k][i])
                    want = str(expected_hashes[k][t, i])
                    assert got == want, (
                        f"obs[{k}][env={i}] diverged at step {t} "
                        f"(done={bool(dones[i])}): got sha1={got[:12]}.., "
                        f"want sha1={want[:12]}.."
                    )

            obs = next_obs
    finally:
        vec_env.close()


@requires_engine
def test_step_passes_terminal_obs_through() -> None:
    """``vec_env.step()`` returns the terminal obs on done (no implicit reset)."""
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    deck_pool = parse_deck_pool([str(deck_path)])
    vec_env = SubprocVecEnv(
        num_envs=2,
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="first",
    )
    try:
        obs = vec_env.reset()
        # Step until at least one env finishes.
        for _ in range(800):
            actions = np.argmax(obs["action_mask"], axis=1).astype(np.int64)
            terminal_obs, _, dones, _ = vec_env.step(actions)
            if dones.any():
                break
            obs = terminal_obs
        else:
            pytest.skip("no done within 800 steps — tune the test")

        # Now: terminal_obs at the done index is the terminal observation.
        # Calling reset_done substitutes new-episode obs at that index.
        post_reset_obs = vec_env.reset_done(dones, terminal_obs)
        done_idx = int(np.where(dones)[0][0])
        # At least ONE obs key must differ between terminal and post-reset.
        # Any-not-all because, e.g., action_mask might happen to coincide
        # by chance even though the new-episode state is different —
        # what we're guarding against is "step() auto-reset leaked back in",
        # in which case ALL keys would be equal between terminal_obs and
        # post_reset_obs.
        differs = any(
            hash_obs_field(terminal_obs[k][done_idx]) != hash_obs_field(post_reset_obs[k][done_idx])
            for k in terminal_obs
        )
        assert differs, (
            f"all obs keys at env={done_idx} are unchanged after reset_done — "
            "auto-reset may have leaked back into step()"
        )
    finally:
        vec_env.close()


@requires_engine
def test_reset_done_no_op_when_no_dones() -> None:
    """``reset_done`` with all-False dones must (a) return the input obs
    untouched and (b) send NO pipe traffic to any worker.  The plan
    explicitly mandates zero traffic — without (b), a future regression
    that always sent a reset and then overwrote with the input could pass
    a hash-only check while silently advancing ``_episode_count`` on
    every worker.
    """
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    deck_pool = parse_deck_pool([str(deck_path)])
    vec_env = SubprocVecEnv(
        num_envs=2,
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="first",
    )
    try:
        obs = vec_env.reset()

        # Wrap each remote's send() with a counter so we can assert
        # zero pipe traffic across the no-op call.
        send_counts = [0] * len(vec_env._remotes)
        original_sends = []
        for i, remote in enumerate(vec_env._remotes):
            original = remote.send
            original_sends.append(original)

            def make_counter(idx, orig):
                def counted(payload):
                    send_counts[idx] += 1
                    return orig(payload)

                return counted

            remote.send = make_counter(i, original)

        try:
            dones = np.zeros(2, dtype=bool)
            result = vec_env.reset_done(dones, obs)
        finally:
            for remote, original in zip(vec_env._remotes, original_sends):
                remote.send = original

        assert send_counts == [0, 0], f"reset_done with all-False dones sent traffic: {send_counts}"
        for k in obs:
            assert np.array_equal(obs[k], result[k]), f"obs[{k}] modified by no-op reset_done"
    finally:
        vec_env.close()


@requires_engine
def test_subproc_vec_env_forwards_pool_handles_to_workers() -> None:
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import SubprocVecEnv
    from yugioh_rl.network import YuGiOhNet
    from yugioh_rl.opponent_pool import OpponentPool

    config = TrainingConfig(self_play=True)
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="greedy",
        network_factory=lambda: YuGiOhNet.from_config(config),
    )

    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    deck_pool = parse_deck_pool([str(deck_path)])

    vec = SubprocVecEnv(
        num_envs=2,
        deck_pool=deck_pool,
        opponent="greedy",
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_config=config,
        seed=42,
    )
    try:
        obs = vec.reset()
        assert obs["action_mask"].shape[0] == 2  # 2 envs
    finally:
        vec.close()
