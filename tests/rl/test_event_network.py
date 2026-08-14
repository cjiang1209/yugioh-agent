import torch

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    EVENT_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
)
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet


def _dummy_obs(B=2):
    cards = torch.zeros(B, MAX_CARDS, CARD_FEATURES, dtype=torch.uint8)
    glob = torch.zeros(B, GLOBAL_FEATURES, dtype=torch.uint8)
    actions = torch.zeros(B, MAX_ACTIONS, ACTION_FEATURES, dtype=torch.uint8)
    mask = torch.ones(B, MAX_ACTIONS, dtype=torch.int8)
    return cards, glob, actions, mask


def test_event_branch_changes_both_heads():
    # Policy-head fusion: events feed BOTH the value AND the policy logits.
    cfg = TrainingConfig(event_history_dim=32)
    net = YuGiOhNet(cfg)
    net.eval()
    cards, glob, actions, mask = _dummy_obs()
    ev0 = torch.zeros(2, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES, dtype=torch.uint8)
    ev1 = ev0.clone()
    ev1[:, -1, 0] = 70  # a chaining event (MSG_CHAINING=70) in newest slot
    ev1[:, -1, 5] = 200  # card code low byte at [5:9]
    with torch.no_grad():
        logits0, val0, _ = net(cards, glob, actions, mask, obs_event=ev0)
        logits1, val1, _ = net(cards, glob, actions, mask, obs_event=ev1)
    assert not torch.allclose(logits0, logits1)  # policy responds to events
    assert not torch.allclose(val0, val1)  # value responds to events


def test_disabled_matches_no_event_arg():
    cfg = TrainingConfig(event_history_dim=0)
    net = YuGiOhNet(cfg)
    assert not hasattr(net, "event_encoder")
    cards, glob, actions, mask = _dummy_obs()
    with torch.no_grad():
        logits, val, _ = net(cards, glob, actions, mask)
    assert logits.shape == (2, MAX_ACTIONS)
    assert val.shape == (2,)
