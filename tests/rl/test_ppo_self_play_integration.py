"""Smoke tests for PPO + self-play wiring."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yugioh_env.server.yugioh_environment")

from tests.rl.conftest import requires_engine


@requires_engine
def test_ppo_with_self_play_constructs_pool(tmp_path) -> None:
    from yugioh_env.opponent import GreedyOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        self_play=True,
        self_play_pool_size=3,
        opponent="greedy",
        num_envs=1,
        total_timesteps=0,
        save_dir=str(tmp_path),
        device="cpu",
    )
    trainer = PPOTrainer(config)
    assert trainer._opponent_pool is not None
    assert isinstance(trainer._opponent_pool._pool[0], GreedyOpponent)
    assert trainer._opponent_pool._shared.pool_size == 3


@requires_engine
def test_ppo_without_self_play_has_no_pool(tmp_path) -> None:
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        self_play=False,
        num_envs=1,
        total_timesteps=0,
        save_dir=str(tmp_path),
        device="cpu",
    )
    trainer = PPOTrainer(config)
    assert trainer._opponent_pool is None
