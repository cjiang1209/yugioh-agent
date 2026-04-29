"""Tests for the actor-learner worker process function."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yugioh_env.deck_parser import parse_ydk
from yugioh_rl.actor_learner import _actor_learner_worker
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.shared_weights import SharedPolicyWeights

from tests.rl.conftest import requires_engine


def _spawn_worker(deck_paths, config: TrainingConfig, rollout_steps: int):
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
        "config_dict": asdict(config),
        "rollout_steps": rollout_steps,
    }
    proc = ctx.Process(target=_actor_learner_worker, kwargs=spawn_kwargs, daemon=True)
    proc.start()
    child.close()
    return proc, parent, shared


@requires_engine
def test_worker_produces_valid_rollout() -> None:
    deck = "assets/decks/blue_eyes.ydk"
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


@requires_engine
def test_worker_refreshes_on_new_version() -> None:
    deck = "assets/decks/blue_eyes.ydk"
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


@requires_engine
def test_worker_handles_shutdown() -> None:
    deck = "assets/decks/blue_eyes.ydk"
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


@requires_engine
def test_worker_resets_on_done() -> None:
    """Worker must reset on done; the obs after a done must differ from the terminal obs."""
    deck = "assets/decks/blue_eyes.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    # 1024 single-env steps comfortably exceeds the ~330-step worst case
    # observed in the captured baseline (4 envs, 3 dones across 338 steps).
    rollout_steps = 1024
    cfg = TrainingConfig(num_envs=1, deck_paths=[deck], rollout_steps=rollout_steps)
    proc, parent, _ = _spawn_worker([deck], cfg, rollout_steps)
    try:
        parent.send(("go", 1))
        cmd, payload = parent.recv()
        assert cmd == "rollout"

        dones = payload["dones"]   # (T,)
        assert dones.any(), (
            "rollout did not cross any done transition; widen rollout_steps"
        )
        obs_cards = payload["obs_cards"]   # (T, 200, 42)

        done_idxs = np.where(dones)[0]
        for t in done_idxs:
            if t + 1 >= rollout_steps:
                continue   # done was the last step, no next obs in window
            terminal_cards = obs_cards[t]
            next_cards = obs_cards[t + 1]
            assert not np.array_equal(terminal_cards, next_cards), (
                f"obs at t+1 ({t + 1}) is byte-identical to terminal obs "
                f"at t ({t}) — worker did not reset on done, the policy is "
                f"being fed a finished-duel obs into the next step"
            )

        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=15)
        if proc.is_alive():
            proc.terminate()


@requires_engine
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_worker_produces_valid_rollout_with_rnn(rnn_type: str) -> None:
    deck = "assets/decks/blue_eyes.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    rollout_steps = 8
    cfg = TrainingConfig(
        num_envs=1,
        deck_paths=[deck],
        rollout_steps=rollout_steps,
        rnn_type=rnn_type,
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        bptt_chunk_len=8,
    )
    proc, parent, _ = _spawn_worker([deck], cfg, rollout_steps)
    try:
        parent.send(("go", 1))
        cmd, payload = parent.recv()
        assert cmd == "rollout"

        for key in ("obs_cards", "obs_global", "obs_actions", "action_mask",
                    "actions", "log_probs", "values", "rewards", "dones",
                    "policy_version", "infos",
                    "final_obs_cards", "final_obs_global",
                    "final_obs_actions", "final_action_mask"):
            assert key in payload, f"missing field {key!r}"

        assert "final_hx" in payload
        final_hx = payload["final_hx"]
        expected_shape = (cfg.rnn_num_layers, 1, cfg.rnn_hidden_dim)
        if rnn_type == "lstm":
            assert isinstance(final_hx, tuple), f"expected (h, c) tuple, got {type(final_hx)}"
            h, c = final_hx
            assert h.shape == expected_shape and c.shape == expected_shape
            assert torch.isfinite(h).all() and torch.isfinite(c).all()
        else:
            assert isinstance(final_hx, torch.Tensor), (
                f"expected single tensor, got {type(final_hx)}"
            )
            assert final_hx.shape == expected_shape
            assert torch.isfinite(final_hx).all()

        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
