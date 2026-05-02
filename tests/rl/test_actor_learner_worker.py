"""Tests for the actor-learner worker process function."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Reuse engine fixtures: skips if libocgcore + cards.cdb absent.
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.shared_weights import SharedPolicyWeights


def _spawn_worker(deck_paths, config: TrainingConfig, rollout_steps: int):
    from yugioh_env.deck_parser import parse_ydk
    from yugioh_rl.actor_learner import _actor_learner_worker

    deck_pool = [parse_ydk(p) for p in deck_paths]

    master = YuGiOhNet.from_config(config)
    shared = SharedPolicyWeights(master)
    shared.publish(master)

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    env_kwargs = {
        "deck_pool": deck_pool,
        "opponent": "random",
        "reward_shaping": False,
        "shaping_lp_weight": 0.0,
        "shaping_card_weight": 0.0,
        "seed": 42,
        "agent_player": "first",
        "opponent_device": "cpu",
    }
    spawn_kwargs = {
        "remote": child,
        "env_kwargs": env_kwargs,
        "weight_handles": shared.share_handles(),
        "config_dict": config.__dict__,
        "rollout_steps": rollout_steps,
    }
    proc = ctx.Process(target=_actor_learner_worker, kwargs=spawn_kwargs, daemon=True)
    proc.start()
    child.close()
    return proc, parent, shared


@pytest.mark.skipif(
    not Path("build/libocgcore.dylib").exists() and not Path("build/libocgcore.so").exists(),
    reason="libocgcore not built (run `make build`)",
)
@pytest.mark.skipif(
    not Path("assets/cards.cdb").exists(),
    reason="assets/cards.cdb not present",
)
def test_worker_produces_valid_rollout() -> None:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    rollout_steps = 8
    cfg = TrainingConfig(num_envs=1, deck_paths=[deck], rollout_steps=rollout_steps)
    proc, parent, _ = _spawn_worker([deck], cfg, rollout_steps)
    try:
        parent.send(("go", 1))
        cmd, payload = parent.recv()
        assert cmd == "rollout"

        T = rollout_steps
        expected_shapes = {
            "obs_cards":   (T, 200, 42),
            "obs_global":  (T, 20),
            "obs_actions": (T, 32, 12),
            "action_mask": (T, 32),
            "actions":     (T,),
            "log_probs":   (T,),
            "values":      (T,),
            "rewards":     (T,),
            "dones":       (T,),
        }
        expected_dtypes = {
            "actions":   np.int64,
            "log_probs": np.float32,
            "values":    np.float32,
            "rewards":   np.float32,
            "dones":     np.bool_,
        }
        for key, shape in expected_shapes.items():
            assert key in payload, f"missing field {key!r}"
            assert payload[key].shape == shape, (
                f"{key} shape {payload[key].shape} != expected {shape}"
            )
        for key, dtype in expected_dtypes.items():
            assert payload[key].dtype == dtype, (
                f"{key} dtype {payload[key].dtype} != expected {dtype}"
            )
        # Numerical sanity: log_probs and values must be finite (NaN here
        # would silently corrupt the trainer's PPO update later).
        assert np.isfinite(payload["log_probs"]).all(), "log_probs has NaN/Inf"
        assert np.isfinite(payload["values"]).all(), "values has NaN/Inf"
        # Sync mode: version is uniform across the rollout → scalar.
        assert isinstance(payload["policy_version"], int)
        assert payload["policy_version"] == 1

        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


@pytest.mark.skipif(
    not Path("build/libocgcore.dylib").exists() and not Path("build/libocgcore.so").exists(),
    reason="libocgcore not built",
)
@pytest.mark.skipif(not Path("assets/cards.cdb").exists(), reason="cards.cdb absent")
def test_worker_refreshes_on_new_version() -> None:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    cfg = TrainingConfig(num_envs=1, deck_paths=[deck], rollout_steps=4)
    proc, parent, shared = _spawn_worker([deck], cfg, 4)
    try:
        # First rollout: version 1 (already published before spawn).
        parent.send(("go", 1))
        cmd, p1 = parent.recv()
        assert cmd == "rollout"
        assert int(p1["policy_version"]) == 1

        # Trainer publishes again → version 2.
        master = YuGiOhNet.from_config(cfg)
        shared.publish(master)

        parent.send(("go", 2))
        cmd, p2 = parent.recv()
        assert cmd == "rollout"
        assert int(p2["policy_version"]) == 2

        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


@pytest.mark.skipif(
    not Path("build/libocgcore.dylib").exists() and not Path("build/libocgcore.so").exists(),
    reason="libocgcore not built",
)
@pytest.mark.skipif(not Path("assets/cards.cdb").exists(), reason="cards.cdb absent")
def test_worker_handles_shutdown() -> None:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    cfg = TrainingConfig(num_envs=1, deck_paths=[deck], rollout_steps=4)
    proc, parent, _ = _spawn_worker([deck], cfg, 4)
    try:
        parent.send(("shutdown", None))
        proc.join(timeout=5)
        assert not proc.is_alive(), "worker did not exit on shutdown"
        assert proc.exitcode == 0, f"worker exited non-zero: {proc.exitcode}"
    finally:
        if proc.is_alive():
            proc.terminate()
