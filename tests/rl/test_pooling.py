"""Tests for board zone-pooling variants (mean / mean_max / attn)."""

from __future__ import annotations

torch = __import__("pytest").importorskip("torch")

from types import SimpleNamespace

import pytest

from yugioh_core.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
)
from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_rl.config import TrainingConfig, normalize_legacy_config
from yugioh_rl.network import _NUM_ZONES, YuGiOhNet
from yugioh_rl.ppo import PPOTrainer

_D = 64  # default card_embed_dim


def _obs(B):
    return (
        torch.zeros(B, MAX_CARDS, CARD_FEATURES, dtype=torch.uint8),
        torch.zeros(B, GLOBAL_FEATURES, dtype=torch.uint8),
        torch.zeros(B, MAX_ACTIONS, ACTION_FEATURES, dtype=torch.uint8),
        torch.ones(B, MAX_ACTIONS, dtype=torch.int8),
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


@pytest.fixture
def attn_net():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="attn"))
    net.eval()
    return net


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
    assert logits.shape == (2, MAX_ACTIONS)
    assert values.shape == (2,)


def test_mean_pool_matches_reference():
    """_pool_zones('mean') equals a hand-rolled masked mean."""
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean"))
    B = 2
    torch.manual_seed(0)
    card_enc = torch.randn(B, MAX_CARDS, _D)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)  # unused by mean pooling
    # Two cards in controller-0 hand (bit 0x02), one in controller-0 mzone (0x04).
    raw_loc[:, 0] = LOCATION_HAND
    raw_loc[:, 1] = LOCATION_HAND
    raw_loc[:, 2] = LOCATION_MZONE

    out = net._pool_zones(card_enc, raw_loc, raw_ctrl, card_ids)
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
    assert logits.shape == (2, MAX_ACTIONS)
    assert values.shape == (2,)


def test_mean_max_width_is_doubled():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean_max"))
    B = 1
    card_enc = torch.zeros(B, MAX_CARDS, _D)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)  # unused by mean_max pooling
    out = net._pool_zones(card_enc, raw_loc, raw_ctrl, card_ids)
    assert out.shape == (B, _NUM_ZONES * _D * 2)


def test_mean_max_max_channel_junk_invariant():
    """Max channel is invariant to junk count; mean channel drifts."""
    net = YuGiOhNet.from_config(TrainingConfig(pooling="mean_max"))
    B = 2
    card_enc = torch.zeros(B, MAX_CARDS, _D)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)  # unused by mean_max pooling

    # Row 0: standout card in ctrl-0 hand, dim-0 activation = 10.
    raw_loc[:, 0] = LOCATION_HAND
    card_enc[:, 0, 0] = 10.0
    # Junk cards, same zone, dim-0 activation = 1.0. Sample 0: 1 junk; sample 1: 5 junk.
    for j in range(1, 6):
        raw_loc[1, j] = LOCATION_HAND
        card_enc[1, j, 0] = 1.0
    raw_loc[0, 1] = LOCATION_HAND
    card_enc[0, 1, 0] = 1.0

    out = net._pool_zones(card_enc, raw_loc, raw_ctrl, card_ids)
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


def test_attn_forward_shapes():
    net = YuGiOhNet.from_config(TrainingConfig(pooling="attn"))
    logits, values, _ = net(*_obs(2))
    assert logits.shape == (2, MAX_ACTIONS)
    assert values.shape == (2,)


def test_attn_head_guard_raises():
    with pytest.raises(ValueError, match="card_embed_dim"):
        YuGiOhNet.from_config(TrainingConfig(pooling="attn", card_embed_dim=50))


def test_attn_and_pool_ignore_padding(attn_net):
    """Padding rows (code 0, location 0) never affect the pooled output."""
    B = 1
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    # One known card in ctrl-0 hand.
    card_ids[:, 0] = 111
    raw_loc[:, 0] = LOCATION_HAND
    torch.manual_seed(1)
    base = torch.randn(B, MAX_CARDS, _D)
    with torch.no_grad():
        out_a = attn_net._pool_zones(base.clone(), raw_loc, raw_ctrl, card_ids)
        # Scribble arbitrary values into padding rows (code 0, loc 0).
        noisy = base.clone()
        noisy[:, 50:] = torch.randn(B, MAX_CARDS - 50, _D)
        out_b = attn_net._pool_zones(noisy, raw_loc, raw_ctrl, card_ids)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_attn_pool_includes_hidden_cards(attn_net):
    """A hidden card (code 0, location != 0) is dropped from attention but
    still contributes to its zone mean."""
    B = 1
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    # Zone 0 (ctrl-0 hand) holds only a hidden card at row 0.
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_loc[:, 0] = LOCATION_HAND  # location set, code stays 0 (hidden)
    card_enc = torch.zeros(B, MAX_CARDS, _D)
    card_enc[:, 0, 3] = 7.0
    with torch.no_grad():
        out = attn_net._pool_zones(card_enc, raw_loc, raw_ctrl, card_ids)
    # Zone-0 mean channel reflects the hidden card (post-LayerNorm,
    # so just assert it is non-zero rather than an exact value).
    assert out[:, 0:_D].abs().sum().item() > 0.0


def test_attn_degenerate_empty_known_no_nan(attn_net):
    """All cards hidden/empty → no NaN in forward (m == 0 short-circuit)."""
    obs_cards, obs_global, obs_actions, action_mask = _obs(2)
    # Give a hidden card (location set, code 0) so location!=0 but code==0.
    obs_cards[:, 0, 4] = LOCATION_HAND  # location byte
    with torch.no_grad():
        logits, values, _ = attn_net(obs_cards, obs_global, obs_actions, action_mask)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(values).any()


def test_attn_mixed_empty_row_no_nan(attn_net):
    """Batch where one row has a known card and another has none exercises
    the per-row all-masked-softmax guard (m > 0, empty_rows non-empty)."""
    B = 2
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_loc = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    raw_ctrl = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    # Sample 0 has a known card; sample 1 has none → m == 1, row 1 all-masked.
    card_ids[0, 0] = 111
    raw_loc[0, 0] = LOCATION_HAND
    card_enc = torch.randn(B, MAX_CARDS, _D)
    with torch.no_grad():
        out = attn_net._pool_zones(card_enc, raw_loc, raw_ctrl, card_ids)
    assert not torch.isnan(out).any()


def test_compat_rejects_attn_arch_mismatch():
    """A mean checkpoint cannot be resumed under an attn config."""
    ckpt = _ckpt(_arch_ns(pooling="mean"))
    with pytest.raises(ValueError, match="pooling"):
        PPOTrainer._validate_checkpoint_compat(TrainingConfig(pooling="attn"), ckpt)


def test_attn_refines_known_cards_behind_hidden(attn_net):
    """Known cards at higher indices than hidden cards must still be refined by
    attention (regression: truncating by known COUNT dropped them)."""
    B = 1
    card_ids = torch.zeros(B, MAX_CARDS, dtype=torch.long)
    # Rows 0-4: hidden (code 0). Rows 5-6: known (code != 0) — behind the hidden run.
    card_ids[0, 5] = 111
    card_ids[0, 6] = 222
    torch.manual_seed(3)
    card_enc = torch.randn(B, MAX_CARDS, _D)
    with torch.no_grad():
        refined = attn_net._attend_cards(card_enc, card_ids)
        norm_only = attn_net.card_attn_norm(card_enc)
    # Known rows must be changed by attention, not left as LayerNorm(card_enc).
    assert not torch.allclose(refined[:, 5:7], norm_only[:, 5:7], atol=1e-5)


def test_from_state_dict_attn_key_guard():
    attn_net = YuGiOhNet.from_config(TrainingConfig(pooling="attn"))
    attn_sd = attn_net.state_dict()
    # attn weights loaded under a mean config → reject with a clear message.
    with pytest.raises(ValueError, match="card_attn"):
        YuGiOhNet.from_state_dict(TrainingConfig(pooling="mean"), attn_sd)

    mean_net = YuGiOhNet.from_config(TrainingConfig(pooling="mean"))
    mean_sd = mean_net.state_dict()
    # mean weights loaded under an attn config → reject.
    with pytest.raises(ValueError, match="card_attn"):
        YuGiOhNet.from_state_dict(TrainingConfig(pooling="attn"), mean_sd)
