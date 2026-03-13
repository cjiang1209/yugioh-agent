"""Tests for --resume training resumption."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.ppo import PPOTrainer


def _make_checkpoint(
    path: str,
    config: TrainingConfig | None = None,
    run_backward: bool = False,
    update: int = 10,
    global_step: int = 1000,
    episode_rewards: list[float] | None = None,
    episode_lengths: list[int] | None = None,
    episode_wins: list[float] | None = None,
    omit_lists: bool = False,
) -> YuGiOhNet:
    """Create a checkpoint and return the network used to build it."""
    if config is None:
        config = TrainingConfig()
    net = YuGiOhNet.from_config(config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    if run_backward:
        cards = torch.zeros(1, 200, 42, dtype=torch.uint8)
        glob = torch.zeros(1, 20, dtype=torch.uint8)
        actions = torch.zeros(1, 32, 12, dtype=torch.uint8)
        mask = torch.ones(1, 32, dtype=torch.int8)
        for _ in range(3):
            logits, value = net(cards, glob, actions, mask)
            loss = -logits.mean() + value.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    data = {
        "update": update,
        "global_step": global_step,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "episode_rewards": episode_rewards or [0.5, -0.5, 1.0],
    }
    if not omit_lists:
        data["episode_lengths"] = episode_lengths or [10, 20, 30]
        data["episode_wins"] = episode_wins or [1.0, 0.0, 1.0]

    torch.save(data, path)
    return net


def test_resume_restores_counters(tmp_path):
    """_resume_update and _resume_global_step should match the checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, update=25, global_step=5000)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._resume_update == 25
    assert trainer._resume_global_step == 5000


def test_resume_restores_weights(tmp_path):
    """Model weights should match the checkpoint exactly."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    config = TrainingConfig(save_dir=str(tmp_path), num_envs=1)
    net = _make_checkpoint(ckpt_path, config)
    ref_param = next(iter(net.parameters())).detach().clone()

    resume_config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(resume_config)
    loaded_param = next(iter(trainer.network.parameters())).detach()

    assert torch.allclose(ref_param, loaded_param), "Weights should match checkpoint"


def test_resume_restores_optimizer_state(tmp_path):
    """Optimizer state should be populated after resume."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, run_backward=True)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert len(trainer.optimizer.state) > 0, "Optimizer state should be populated"


def test_resume_restores_episode_tracking(tmp_path):
    """Episode tracking lists should be restored from checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    rewards = [1.0, -1.0, 0.5]
    lengths = [15, 25, 35]
    wins = [1.0, 0.0, 1.0]
    _make_checkpoint(
        ckpt_path,
        episode_rewards=rewards,
        episode_lengths=lengths,
        episode_wins=wins,
    )

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._episode_rewards == rewards
    assert trainer._episode_lengths == lengths
    assert trainer._episode_wins == wins


def test_resume_backward_compat_missing_lists(tmp_path):
    """Old checkpoints without episode_lengths/wins should resume with empty lists."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, omit_lists=True)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._episode_rewards == [0.5, -0.5, 1.0]  # present in checkpoint
    assert trainer._episode_lengths == []  # missing → empty fallback
    assert trainer._episode_wins == []  # missing → empty fallback


def test_resume_arch_mismatch_rejected(tmp_path):
    """Architecture mismatch should raise ValueError."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
        card_embed_dim=128,
    )
    with pytest.raises(ValueError, match="Architecture mismatch"):
        PPOTrainer(config)


def test_resume_infers_save_dir(tmp_path):
    """save_dir should be the checkpoint's parent directory."""
    run_dir = tmp_path / "runs" / "20260101_120000_seed42"
    run_dir.mkdir(parents=True)
    ckpt_path = str(run_dir / "checkpoint_10.pt")
    _make_checkpoint(ckpt_path)

    config = TrainingConfig(
        save_dir=str(run_dir),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer.config.save_dir == str(run_dir)


def test_resume_lr_override(tmp_path):
    """Optimizer LR should use the CLI value, not the checkpoint's stale LR."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    old_config = TrainingConfig(learning_rate=1e-3)
    _make_checkpoint(ckpt_path, config=old_config, run_backward=True)

    new_lr = 5e-5
    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
        learning_rate=new_lr,
    )
    trainer = PPOTrainer(config)

    for pg in trainer.optimizer.param_groups:
        assert pg["lr"] == new_lr, f"Expected LR {new_lr}, got {pg['lr']}"


def test_resume_early_return_when_training_complete(tmp_path, caplog):
    """train() should return immediately if resume update >= num_updates."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    # Checkpoint at update=10, global_step=20480 (10 * 256 * 8)
    _make_checkpoint(ckpt_path, update=10, global_step=20480)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=8,
        rollout_steps=256,
        # total_timesteps that yields exactly 10 updates → already done
        total_timesteps=256 * 8 * 10,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    import logging
    with caplog.at_level(logging.WARNING):
        trainer.train()

    assert "training already complete" in caplog.text
