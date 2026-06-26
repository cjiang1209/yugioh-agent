"""Tests for pending chain feature decoding and network integration."""

from __future__ import annotations

torch = __import__("pytest").importorskip("torch")

from yugioh_core.encoding import CHAIN_ENTRY_FEATURES, MAX_PENDING_CHAIN, encode_chain_entry
from yugioh_rl.config import TrainingConfig
from yugioh_rl.features import CHAIN_FEAT_DIM, decode_pending_chain


def test_decode_pending_chain_shapes():
    """decode_pending_chain returns correct shapes."""
    B = 4
    raw = torch.zeros(B, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES, dtype=torch.uint8)
    codes, desc_p, desc_n, feats = decode_pending_chain(raw)
    assert codes.shape == (B, MAX_PENDING_CHAIN)
    assert desc_p.shape == (B, MAX_PENDING_CHAIN)
    assert desc_n.shape == (B, MAX_PENDING_CHAIN)
    assert feats.shape == (B, MAX_PENDING_CHAIN, CHAIN_FEAT_DIM)


def test_decode_pending_chain_values():
    """decode_pending_chain correctly extracts fields from encoded entry."""
    entry = encode_chain_entry(
        code=44444,
        desc=(0x1234 << 20) | 7,
        controller=1,
        location=0x04,
        sequence=3,
        chain_link=2,
    )
    raw = torch.zeros(1, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES, dtype=torch.uint8)
    raw[0, 0] = torch.from_numpy(entry)

    codes, desc_p, desc_n, feats = decode_pending_chain(raw)
    assert codes[0, 0].item() == 44444
    assert desc_p[0, 0].item() == 0x1234
    assert desc_n[0, 0].item() == 7
    assert feats[0, 0, 0].item() == 1.0  # controller
    assert feats[0, 0, 9].item() == 2.0  # chain_link


def test_network_forward_with_chain_disabled():
    """chain_embed_dim=0: network forward works without obs_chain."""
    config = TrainingConfig(chain_embed_dim=0)
    from yugioh_rl.network import YuGiOhNet

    net = YuGiOhNet.from_config(config)
    B = 2
    obs_cards = torch.zeros(B, 200, 42, dtype=torch.uint8)
    obs_global = torch.zeros(B, 20, dtype=torch.uint8)
    obs_actions = torch.zeros(B, 32, 28, dtype=torch.uint8)
    action_mask = torch.ones(B, 32, dtype=torch.int8)
    logits, values, hx = net(obs_cards, obs_global, obs_actions, action_mask)
    assert logits.shape == (B, 32)
    assert values.shape == (B,)


def test_network_forward_with_chain_enabled():
    """chain_embed_dim>0: network forward uses obs_chain."""
    config = TrainingConfig(chain_embed_dim=32)
    from yugioh_rl.network import YuGiOhNet

    net = YuGiOhNet.from_config(config)
    B = 2
    obs_cards = torch.zeros(B, 200, 42, dtype=torch.uint8)
    obs_global = torch.zeros(B, 20, dtype=torch.uint8)
    obs_actions = torch.zeros(B, 32, 28, dtype=torch.uint8)
    action_mask = torch.ones(B, 32, dtype=torch.int8)
    obs_chain = torch.zeros(B, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES, dtype=torch.uint8)
    logits, values, hx = net(obs_cards, obs_global, obs_actions, action_mask, obs_chain=obs_chain)
    assert logits.shape == (B, 32)
    assert values.shape == (B,)


def test_network_chain_changes_output():
    """Non-zero chain entries produce different logits than all-zero chain."""
    config = TrainingConfig(chain_embed_dim=32)
    from yugioh_rl.network import YuGiOhNet

    net = YuGiOhNet.from_config(config)
    net.eval()
    B = 1
    obs_cards = torch.zeros(B, 200, 42, dtype=torch.uint8)
    obs_global = torch.zeros(B, 20, dtype=torch.uint8)
    obs_actions = torch.zeros(B, 32, 28, dtype=torch.uint8)
    action_mask = torch.ones(B, 32, dtype=torch.int8)

    empty_chain = torch.zeros(B, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES, dtype=torch.uint8)
    entry = encode_chain_entry(
        code=12345, desc=0, controller=1, location=0x04, sequence=0, chain_link=1
    )
    nonempty_chain = empty_chain.clone()
    nonempty_chain[0, 0] = torch.from_numpy(entry)

    with torch.no_grad():
        logits_empty, _, _ = net(
            obs_cards, obs_global, obs_actions, action_mask, obs_chain=empty_chain
        )
        logits_full, _, _ = net(
            obs_cards, obs_global, obs_actions, action_mask, obs_chain=nonempty_chain
        )
    assert not torch.allclose(logits_empty, logits_full), "Chain entries should affect output"
