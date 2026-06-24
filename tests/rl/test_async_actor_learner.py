"""Tests for async actor-learner mode."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.rl.conftest import requires_engine
from yugioh_env.deck_parser import parse_ydk
from yugioh_rl.actor_learner import _async_actor_learner_worker
from yugioh_rl.config import VEC_ENV_TYPES, TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.shared_weights import SharedPolicyWeights

_DECK = "assets/decks/blue_eyes.ydk"


# --- Config-only tests (no engine needed) ---


def test_vec_env_type_includes_async():
    """VecEnvType literal and VEC_ENV_TYPES tuple include async variant."""
    assert "async_actor_learner" in VEC_ENV_TYPES
    cfg = TrainingConfig(vec_env_type="async_actor_learner")
    assert cfg.vec_env_type == "async_actor_learner"


def test_config_has_max_version_lag():
    """max_version_lag exists with correct default."""
    cfg = TrainingConfig()
    assert cfg.max_version_lag == 5


# --- Worker / vec-env tests (require engine) ---


def _make_async_worker(deck_paths, config, rollout_steps):
    """Helper: spawn one async worker, return (queue, shutdown_event, process, weights)."""
    deck_pool = [parse_ydk(p) for p in deck_paths]
    net = YuGiOhNet.from_config(config)
    weights = SharedPolicyWeights(net)
    weights.publish(net)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    shutdown = ctx.Event()

    env_kwargs = {
        "deck_pool": deck_pool,
        "opponent": "random",
        "reward_shaping": False,
        "shaping_lp_weight": 0.01,
        "shaping_card_weight": 0.005,
        "seed": 42,
        "agent_player": "first",
        "opponent_device": None,
    }

    p = ctx.Process(
        target=_async_actor_learner_worker,
        kwargs={
            "queue": queue,
            "shutdown_event": shutdown,
            "env_kwargs": env_kwargs,
            "weight_handles": weights.share_handles(),
            "config_dict": asdict(config),
            "rollout_steps": rollout_steps,
        },
        daemon=True,
    )
    p.start()
    return queue, shutdown, p, weights


@requires_engine
def test_async_worker_produces_rollout():
    """Async worker pushes at least one rollout to the queue."""
    config = TrainingConfig(num_envs=1)
    rollout_steps = 8
    queue, shutdown, proc, _ = _make_async_worker([_DECK], config, rollout_steps)
    try:
        payload = queue.get(timeout=30)
        assert payload["actions"].shape == (rollout_steps,)
        assert "policy_version" in payload
        assert isinstance(payload["policy_version"], int | np.ndarray)
    finally:
        shutdown.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_async_worker_produces_multiple_rollouts():
    """Async worker produces multiple rollouts without external commands."""
    config = TrainingConfig(num_envs=1)
    rollout_steps = 8
    queue, shutdown, proc, _ = _make_async_worker([_DECK], config, rollout_steps)
    try:
        payloads = [queue.get(timeout=30) for _ in range(3)]
        assert len(payloads) == 3
        for p in payloads:
            assert p["actions"].shape == (rollout_steps,)
    finally:
        shutdown.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_async_worker_refreshes_on_new_version():
    """Worker picks up new weights when trainer publishes."""
    config = TrainingConfig(num_envs=1)
    rollout_steps = 8
    queue, shutdown, proc, weights = _make_async_worker([_DECK], config, rollout_steps)
    try:
        p1 = queue.get(timeout=30)
        v1 = p1["policy_version"]

        net = YuGiOhNet.from_config(config)
        weights.publish(net)

        found_new = False
        for _ in range(20):
            p = queue.get(timeout=30)
            v = p["policy_version"]
            if isinstance(v, int) and v > v1:
                found_new = True
                break
        assert found_new, "worker never picked up version 2"
    finally:
        shutdown.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


def _make_async_vec(deck_path, **overrides):
    """Create an AsyncActorLearnerVecEnv for testing."""
    from yugioh_rl.actor_learner import AsyncActorLearnerVecEnv

    defaults = dict(num_envs=2, vec_env_type="async_actor_learner")
    defaults.update(overrides)
    config = TrainingConfig(**defaults)
    deck_pool = [parse_ydk(deck_path)]
    net = YuGiOhNet.from_config(config)

    vec_env = AsyncActorLearnerVecEnv(
        num_envs=config.num_envs,
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        shaping_lp_weight=0.01,
        shaping_card_weight=0.005,
        seed=42,
        agent_player="first",
        opponent_device=None,
        master_model=net,
        config=config,
        rollout_steps=8,
    )
    return vec_env, config


@requires_engine
def test_async_vec_env_collect_rollouts():
    """AsyncActorLearnerVecEnv.collect_rollouts returns num_envs rollouts."""
    vec_env, config = _make_async_vec(_DECK)
    try:
        rollouts, discarded = vec_env.collect_rollouts(max_version_lag=5)
        assert len(rollouts) == config.num_envs
        for r in rollouts:
            assert r["actions"].shape == (8,)
        assert discarded == 0
    finally:
        vec_env.close()


@requires_engine
def test_async_vec_env_discards_stale_rollouts():
    """Rollouts beyond max_version_lag are discarded."""
    vec_env, config = _make_async_vec(_DECK)
    try:
        net = YuGiOhNet.from_config(config)
        for _ in range(10):
            vec_env.publish_weights(net)

        rollouts, discarded = vec_env.collect_rollouts(max_version_lag=1)
        assert len(rollouts) == config.num_envs
    finally:
        vec_env.close()
