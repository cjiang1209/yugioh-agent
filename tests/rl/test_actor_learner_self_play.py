"""Smoke test: ActorLearnerVecEnv accepts opponent_pool kwargs."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yugioh_env.server.yugioh_environment")

from tests.rl.conftest import requires_engine
from yugioh_rl.actor_learner import ActorLearnerVecEnv
from yugioh_rl.config import TrainingConfig
from yugioh_rl.env_wrapper import parse_deck_pool
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.opponent_pool import OpponentPool


@requires_engine
def test_actor_learner_vec_env_forwards_pool_handles_to_workers() -> None:
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    deck_pool = parse_deck_pool([str(deck_path)])

    config = TrainingConfig(self_play=True, vec_env_type="sync_actor_learner")
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="greedy",
        network_factory=lambda: YuGiOhNet.from_config(config),
    )
    master = YuGiOhNet.from_config(config)

    vec = ActorLearnerVecEnv(
        num_envs=2,
        deck_pool=deck_pool,
        opponent="greedy",
        reward_shaping=True,
        shaping_lp_weight=0.01,
        shaping_card_weight=0.005,
        seed=42,
        agent_player="random",
        opponent_device=None,
        master_model=master,
        config=config,
        rollout_steps=4,
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_config=config,
        worker_timeout_s=30.0,
    )
    try:
        rollouts = vec.collect_rollouts()
        assert len(rollouts) == 2
        for r in rollouts:
            assert r["actions"].shape == (4,)
    finally:
        vec.close()
