"""Tests for --init-checkpoint weight initialization."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.ppo import PPOTrainer


def _make_checkpoint(path: str, config: TrainingConfig | None = None,
                     run_backward: bool = False) -> YuGiOhNet:
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
            logits, value, _ = net(cards, glob, actions, mask)
            loss = -logits.mean() + value.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({
        "update": 10,
        "global_step": 1000,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)
    return net


def test_init_checkpoint_loads_weights(tmp_path):
    """Weights from checkpoint should be loaded, not randomly initialized."""
    ckpt_path = str(tmp_path / "ckpt.pt")

    # Create checkpoint and grab a known weight tensor
    config = TrainingConfig(save_dir=str(tmp_path / "run0"), num_envs=1)
    net = _make_checkpoint(ckpt_path, config)
    ref_param = next(iter(net.parameters())).detach().clone()

    # Init PPOTrainer from checkpoint
    init_config = TrainingConfig(
        save_dir=str(tmp_path / "run1"),
        num_envs=1,
        init_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(init_config)
    loaded_param = next(iter(trainer.network.parameters())).detach()

    assert torch.allclose(ref_param, loaded_param), "Weights should match checkpoint"


def test_init_checkpoint_arch_mismatch_rejected(tmp_path):
    """Mismatched architecture dims should raise ValueError."""
    ckpt_path = str(tmp_path / "ckpt.pt")

    # Checkpoint with default card_embed_dim=64
    _make_checkpoint(ckpt_path)

    # Try to load with card_embed_dim=128
    config = TrainingConfig(
        save_dir=str(tmp_path / "run"),
        num_envs=1,
        init_checkpoint=ckpt_path,
        card_embed_dim=128,
    )
    with pytest.raises(ValueError, match="Architecture mismatch"):
        PPOTrainer(config)


def test_init_checkpoint_text_mode_mismatch_rejected(tmp_path):
    """Loading symbolic checkpoint with --card-embeddings should raise ValueError."""
    ckpt_path = str(tmp_path / "ckpt.pt")

    # Symbolic-mode checkpoint (no text embeddings)
    _make_checkpoint(ckpt_path)

    config = TrainingConfig(
        save_dir=str(tmp_path / "run"),
        num_envs=1,
        init_checkpoint=ckpt_path,
        card_embeddings_path="fake.pt",
    )
    with pytest.raises(ValueError, match="text embedding"):
        PPOTrainer(config)


def test_resume_optimizer_loads_state(tmp_path):
    """Optimizer state should be populated when resume_optimizer=True."""
    ckpt_path = str(tmp_path / "ckpt.pt")

    _make_checkpoint(ckpt_path, run_backward=True)

    config = TrainingConfig(
        save_dir=str(tmp_path / "run"),
        num_envs=1,
        init_checkpoint=ckpt_path,
        resume_optimizer=True,
    )
    trainer = PPOTrainer(config)

    # Optimizer state should have entries (not empty like a fresh optimizer)
    assert len(trainer.optimizer.state) > 0, "Optimizer state should be populated"
