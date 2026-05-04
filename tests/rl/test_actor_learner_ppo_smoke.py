"""End-to-end smoke test: PPOTrainer with vec_env_type=sync_actor_learner."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.ppo import PPOTrainer

from tests.rl.conftest import requires_engine


@requires_engine
def test_actor_learner_ppo_runs_to_completion(tmp_path) -> None:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    cfg = TrainingConfig(
        num_envs=2,
        deck_paths=[deck],
        opponent="random",
        reward_shaping=False,
        rollout_steps=8,
        num_epochs=1,
        minibatch_size=8,
        total_timesteps=64,
        eval_interval=999,
        save_interval=999,
        log_interval=1,
        save_dir=str(tmp_path),
        vec_env_type="sync_actor_learner",
        device="cpu",
    )
    trainer = PPOTrainer(cfg)
    weights_before = {
        name: p.detach().clone()
        for name, p in trainer.network.state_dict().items()
        if p.dtype.is_floating_point
    }
    trainer.train()

    weights_after = trainer.network.state_dict()
    nonfinite = [
        name for name, p in weights_after.items()
        if p.dtype.is_floating_point and not torch.isfinite(p).all()
    ]
    assert not nonfinite, f"non-finite params: {nonfinite}"

    # At least one trainable param must have changed — otherwise gradients
    # never reached the network (e.g. publish_weights silently no-op'd, or
    # the optimizer step skipped).
    changed = [
        name for name, before in weights_before.items()
        if not torch.equal(before, weights_after[name])
    ]
    assert changed, "no parameters changed during training — optimizer step is dead"


@requires_engine
def test_actor_learner_ppo_runs_with_rnn(tmp_path) -> None:
    """End-to-end smoke: actor-learner + LSTM + TBPTT update."""
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    cfg = TrainingConfig(
        num_envs=2,
        deck_paths=[deck],
        opponent="random",
        reward_shaping=False,
        rollout_steps=8,
        bptt_chunk_len=8,
        num_epochs=1,
        minibatch_size=8,
        total_timesteps=64,
        eval_interval=999,
        save_interval=999,
        log_interval=1,
        save_dir=str(tmp_path),
        vec_env_type="sync_actor_learner",
        rnn_type="lstm",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        device="cpu",
    )
    trainer = PPOTrainer(cfg)
    weights_before = {
        name: p.detach().clone()
        for name, p in trainer.network.state_dict().items()
        if p.dtype.is_floating_point
    }
    trainer.train()

    weights_after = trainer.network.state_dict()
    nonfinite = [
        name for name, p in weights_after.items()
        if p.dtype.is_floating_point and not torch.isfinite(p).all()
    ]
    assert not nonfinite, f"non-finite params: {nonfinite}"

    changed = [
        name for name, before in weights_before.items()
        if not torch.equal(before, weights_after[name])
    ]
    assert changed, "no parameters changed during training — optimizer step is dead"
