"""Tests for board zone-pooling variants (mean / mean_max / attn)."""

from __future__ import annotations

torch = __import__("pytest").importorskip("torch")

from types import SimpleNamespace

import pytest

from yugioh_rl.config import TrainingConfig, normalize_legacy_config
from yugioh_rl.network import _NUM_ZONES, YuGiOhNet
from yugioh_rl.ppo import PPOTrainer

_D = 64  # default card_embed_dim


def _obs(B):
    return (
        torch.zeros(B, 200, 42, dtype=torch.uint8),
        torch.zeros(B, 20, dtype=torch.uint8),
        torch.zeros(B, 32, 28, dtype=torch.uint8),
        torch.ones(B, 32, dtype=torch.int8),
    )


def _arch_ns(**overrides):
    """A minimal ckpt-config namespace with all arch fields present."""
    base = dict(
        card_embed_dim=64,
        global_embed_dim=64,
        board_hidden_dim=256,
        action_embed_dim=64,
        text_embed_dim=64,
        learned_embed_dim=8,
        rnn_type="none",
        chain_embed_dim=32,
        event_history_dim=0,
        pooling="mean",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _ckpt(config):
    """Wrap a config namespace as a checkpoint dict. _validate_checkpoint_compat
    reads ckpt["model_state_dict"] (for text-embedding detection) on the
    no-mismatch path, so the fixture must provide it (empty = symbolic mode)."""
    return {"config": config, "model_state_dict": {}}


def test_pooling_defaults_to_mean():
    assert TrainingConfig().pooling == "mean"


def test_normalize_legacy_backfills_pooling():
    cfg = TrainingConfig()
    del cfg.__dict__["pooling"]  # simulate a pre-feature pickled config
    normalize_legacy_config(cfg)
    assert cfg.pooling == "mean"


def test_mean_forward_shapes():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean"))
    logits, values, _ = net(*_obs(2))
    assert logits.shape == (2, 32)
    assert values.shape == (2,)


def test_mean_pool_matches_reference():
    """_pool_zones('mean') equals a hand-rolled masked mean."""
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean"))
    B = 2
    torch.manual_seed(0)
    card_enc = torch.randn(B, 200, _D)
    raw_loc = torch.zeros(B, 200, dtype=torch.long)
    raw_ctrl = torch.zeros(B, 200, dtype=torch.long)
    # Two cards in controller-0 hand (bit 0x02), one in controller-0 mzone (0x04).
    raw_loc[:, 0] = 0x02
    raw_loc[:, 1] = 0x02
    raw_loc[:, 2] = 0x04

    out = net._pool_zones(card_enc, raw_loc, raw_ctrl)
    assert out.shape == (B, _NUM_ZONES * _D)  # mult == 1

    # Zone 0 = (ctrl 0, hand): mean of rows 0 and 1.
    expected_hand = (card_enc[:, 0] + card_enc[:, 1]) / 2
    assert torch.allclose(out[:, 0:_D], expected_hand, atol=1e-6)
    # Zone 1 = (ctrl 0, mzone): just row 2.
    assert torch.allclose(out[:, _D : 2 * _D], card_enc[:, 2], atol=1e-6)
    # Zone 2 = (ctrl 0, szone): empty → zeros.
    assert torch.allclose(out[:, 2 * _D : 3 * _D], torch.zeros(B, _D), atol=1e-6)


def test_mean_max_forward_shapes():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean_max"))
    logits, values, _ = net(*_obs(2))
    assert logits.shape == (2, 32)
    assert values.shape == (2,)


def test_mean_max_width_is_doubled():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean_max"))
    B = 1
    card_enc = torch.zeros(B, 200, _D)
    raw_loc = torch.zeros(B, 200, dtype=torch.long)
    raw_ctrl = torch.zeros(B, 200, dtype=torch.long)
    out = net._pool_zones(card_enc, raw_loc, raw_ctrl)
    assert out.shape == (B, _NUM_ZONES * _D * 2)


def test_mean_max_max_channel_junk_invariant():
    """Max channel is invariant to junk count; mean channel drifts."""
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean_max"))
    B = 2
    card_enc = torch.zeros(B, 200, _D)
    raw_loc = torch.zeros(B, 200, dtype=torch.long)
    raw_ctrl = torch.zeros(B, 200, dtype=torch.long)

    # Row 0: standout card in ctrl-0 hand, dim-0 activation = 10.
    raw_loc[:, 0] = 0x02
    card_enc[:, 0, 0] = 10.0
    # Junk cards, same zone, dim-0 activation = 1.0. Sample 0: 1 junk; sample 1: 5 junk.
    for j in range(1, 6):
        raw_loc[1, j] = 0x02
        card_enc[1, j, 0] = 1.0
    raw_loc[0, 1] = 0x02
    card_enc[0, 1, 0] = 1.0

    out = net._pool_zones(card_enc, raw_loc, raw_ctrl)
    # Zone 0 layout: [mean(0:D), max(D:2D)].
    mean_dim0 = out[:, 0]
    max_dim0 = out[:, _D]
    # Max channel: identical across the two junk counts.
    assert torch.allclose(max_dim0[0], max_dim0[1], atol=1e-6)
    assert abs(max_dim0[0].item() - 10.0) < 1e-6
    # Mean channel: drifts with junk count → (10+1)/2=5.5 vs (10+5)/6≈2.5.
    assert abs(mean_dim0[0].item() - 5.5) < 1e-6
    assert abs(mean_dim0[1].item() - 2.5) < 1e-6
    assert not torch.allclose(mean_dim0[0], mean_dim0[1])


def test_compat_rejects_pooling_mismatch():
    ckpt = _ckpt(_arch_ns(pooling="mean"))
    cli = TrainingConfig(pooling="mean_max")
    with pytest.raises(ValueError, match="pooling"):
        PPOTrainer._validate_checkpoint_compat(cli, ckpt)


def test_compat_legacy_missing_pooling_is_mean():
    legacy = _arch_ns()
    del legacy.pooling  # pre-feature checkpoint: field absent
    ckpt = _ckpt(legacy)
    # Treated as "mean": a mean run is accepted...
    PPOTrainer._validate_checkpoint_compat(TrainingConfig(pooling="mean"), ckpt)
    # ...a mean_max run is rejected.
    with pytest.raises(ValueError, match="pooling"):
        PPOTrainer._validate_checkpoint_compat(TrainingConfig(pooling="mean_max"), ckpt)
