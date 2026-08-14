"""Tests for NetworkOpponent stochastic-sampling mode."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from tests.env.conftest import obs_from_action_count
from yugioh_core.encoding import MAX_ACTIONS
from yugioh_env.opponent import NetworkOpponent


class _FakeNet(nn.Module):
    """Minimal stand-in: ignores inputs, returns canned logits + dummy value + hx.

    ``logits`` covers only the legal actions under test; padded out to
    ``MAX_ACTIONS`` with filler values for slots the encoded mask marks
    illegal, so they're masked to -inf regardless of the filler.
    """

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        padded = list(logits) + [0.0] * (MAX_ACTIONS - len(logits))
        self._logits = torch.tensor([padded], dtype=torch.float32)

    def init_hx(self, batch_size: int, device):
        return None

    def forward(
        self,
        obs_cards,
        obs_global,
        obs_actions,
        action_mask,
        hx=None,
        obs_chain=None,
        obs_event=None,
    ):
        return self._logits, torch.zeros(1, 1), hx


def test_stochastic_false_is_argmax() -> None:
    net = _FakeNet([3.0, 1.0, 0.5])
    opp = NetworkOpponent(net, stochastic=False)
    action, _ = opp.select_action(obs_from_action_count(3))
    assert action == 0


def test_stochastic_true_samples_distribution() -> None:
    # Logits where action 2 is much more likely; with temperature 1 we
    # should see it dominate over many samples.
    net = _FakeNet([0.0, 0.0, 5.0])
    obs = obs_from_action_count(3)
    counts = [0, 0, 0]
    for seed in range(200):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        action, _ = opp.select_action(obs)
        counts[action] += 1
    assert counts[2] > counts[0]
    assert counts[2] > counts[1]
    assert counts[2] > 100  # vast majority


def test_stochastic_low_temperature_approaches_argmax() -> None:
    # Temperature very low (0.01) effectively == argmax.
    net = _FakeNet([3.0, 1.0, 0.5])
    torch.manual_seed(0)
    opp = NetworkOpponent(net, stochastic=True, temperature=0.01)
    action, _ = opp.select_action(obs_from_action_count(3))
    assert action == 0


def test_stochastic_temperature_zero_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=0.0)


def test_stochastic_negative_temperature_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=-0.5)


def test_stochastic_respects_action_mask() -> None:
    # The highest logit sits on the one illegal slot, so -inf masking alone
    # has to keep sampling inside the legal prefix. Legality is a descriptor
    # count, so only a prefix is expressible: an illegal slot *between* legal
    # ones -- which would also catch an index-clamping bug -- cannot be built.
    net = _FakeNet([0.0, 0.0, 10.0])
    obs = obs_from_action_count(2)
    for seed in range(50):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        a, _ = opp.select_action(obs)
        assert a in (0, 1), f"illegal action {a} was sampled at seed {seed}"
