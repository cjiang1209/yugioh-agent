"""Integration tests for ActorLearnerVecEnv."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yugioh_env.deck_parser import parse_ydk
from yugioh_rl.actor_learner import ActorLearnerVecEnv
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet

from tests.rl.conftest import requires_engine


@pytest.fixture
def starter_deck() -> str:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")
    return deck


def _make_vec(starter_deck: str, **overrides) -> tuple[ActorLearnerVecEnv, TrainingConfig]:
    """Build a small ActorLearnerVecEnv suitable for integration tests."""
    cfg = TrainingConfig(
        num_envs=overrides.pop("num_envs", 2),
        deck_paths=[starter_deck],
        rollout_steps=overrides.pop("rollout_steps", 4),
        opponent="random",
        reward_shaping=False,
    )
    deck_pool = [parse_ydk(starter_deck)]
    master = YuGiOhNet.from_config(cfg)
    vec = ActorLearnerVecEnv(
        num_envs=cfg.num_envs,
        deck_pool=deck_pool,
        opponent=cfg.opponent,
        reward_shaping=cfg.reward_shaping,
        shaping_lp_weight=cfg.shaping_lp_weight,
        shaping_card_weight=cfg.shaping_card_weight,
        seed=cfg.seed,
        agent_player=cfg.agent_player,
        opponent_device="cpu",
        master_model=master,
        config=cfg,
        rollout_steps=cfg.rollout_steps,
    )
    return vec, cfg


@requires_engine
def test_collect_rollouts_returns_n_rollouts(starter_deck: str) -> None:
    vec, cfg = _make_vec(starter_deck)
    try:
        rollouts = vec.collect_rollouts()
        assert len(rollouts) == cfg.num_envs
        for r in rollouts:
            assert r["actions"].shape == (cfg.rollout_steps,)
            assert int(r["policy_version"]) == 1
    finally:
        vec.close()


@requires_engine
def test_collect_rollouts_multiple_cycles(starter_deck: str) -> None:
    """Two consecutive collect_rollouts calls succeed and don't leak pipe state."""
    vec, cfg = _make_vec(starter_deck)
    try:
        first = vec.collect_rollouts()
        assert len(first) == cfg.num_envs

        second = vec.collect_rollouts()
        assert len(second) == cfg.num_envs
        for r in second:
            assert r["actions"].shape == (cfg.rollout_steps,)
    finally:
        vec.close()


@requires_engine
def test_publish_weights_propagates_to_workers(starter_deck: str) -> None:
    """publish_weights bumps the version; workers tag the next rollout."""
    vec, cfg = _make_vec(starter_deck)
    try:
        first = vec.collect_rollouts()
        assert all(int(r["policy_version"]) == 1 for r in first)

        master = YuGiOhNet.from_config(cfg)
        new_version = vec.publish_weights(master)
        assert new_version == 2

        second = vec.collect_rollouts()
        assert all(int(r["policy_version"]) == 2 for r in second)
    finally:
        vec.close()


@requires_engine
def test_rollout_includes_final_obs_and_infos(starter_deck: str) -> None:
    vec, cfg = _make_vec(starter_deck)
    try:
        rollouts = vec.collect_rollouts()
        for r in rollouts:
            for k in ("final_obs_cards", "final_obs_global",
                      "final_obs_actions", "final_action_mask"):
                assert k in r
            assert r["final_obs_cards"].shape == r["obs_cards"][0].shape
            assert "infos" in r
            assert isinstance(r["infos"], list)
            assert len(r["infos"]) == cfg.rollout_steps
    finally:
        vec.close()
