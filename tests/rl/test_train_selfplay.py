"""End-to-end smoke test: short PPO run with self-play."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.rl.conftest import requires_engine


@requires_engine
def test_short_selfplay_run_completes(tmp_path) -> None:
    """Run a tiny PPO loop with self-play and verify the pool accumulates snapshots."""
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        self_play=True,
        self_play_pool_size=3,
        self_play_temperature=1.0,
        opponent="greedy",
        num_envs=1,
        total_timesteps=64,
        rollout_steps=64,
        minibatch_size=32,
        num_epochs=1,
        save_interval=1,
        eval_interval=10_000,
        save_dir=str(tmp_path),
        device="cpu",
    )
    trainer = PPOTrainer(config)
    trainer.train()

    # After training, the pool must have at least one snapshot beyond
    # the initial seed.
    assert trainer._opponent_pool._shared.total_adds >= 2
