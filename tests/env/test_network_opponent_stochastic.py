"""Tests for NetworkOpponent stochastic-sampling mode."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import torch.nn as nn

from yugioh_core.encoding import MAX_ACTIONS
from yugioh_env.models import YuGiOhObservation
from yugioh_env.opponent import NetworkOpponent


class _FakeNet(nn.Module):
    """Minimal stand-in: ignores inputs, returns canned logits + dummy value + hx.

    ``logits`` covers only the legal actions under test; padded out to
    ``MAX_ACTIONS`` with filler values that ``_make_obs``'s mask marks
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


def _make_obs(n_actions: int) -> YuGiOhObservation:
    """First `n_actions` slots legal, the rest (padding to MAX_ACTIONS) illegal."""
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[:n_actions] = 1
    return YuGiOhObservation(action_mask=mask)


def test_stochastic_false_is_argmax() -> None:
    net = _FakeNet([3.0, 1.0, 0.5])
    opp = NetworkOpponent(net, stochastic=False)
    assert opp.select_action(_make_obs(3)) == 0


def test_stochastic_true_samples_distribution() -> None:
    # Logits where action 2 is much more likely; with temperature 1 we
    # should see it dominate over many samples.
    net = _FakeNet([0.0, 0.0, 5.0])
    obs = _make_obs(3)
    counts = [0, 0, 0]
    for seed in range(200):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        counts[opp.select_action(obs)] += 1
    assert counts[2] > counts[0]
    assert counts[2] > counts[1]
    assert counts[2] > 100  # vast majority


def test_stochastic_low_temperature_approaches_argmax() -> None:
    # Temperature very low (0.01) effectively == argmax.
    net = _FakeNet([3.0, 1.0, 0.5])
    torch.manual_seed(0)
    opp = NetworkOpponent(net, stochastic=True, temperature=0.01)
    assert opp.select_action(_make_obs(3)) == 0


def test_stochastic_temperature_zero_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=0.0)


def test_stochastic_negative_temperature_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=-0.5)


def test_stochastic_respects_action_mask() -> None:
    # Highest logit at action 0, but mask it out (sparse mask: only 1 and 2
    # are legal, action 0 is not, even though mask.sum() == 2 != 3). This
    # deliberately exercises a case where a naive `min(action, mask.sum()-1)`
    # clamp would be WRONG -- the -inf masking alone must fully constrain
    # sampling to the legal indices {1, 2}.
    net = _FakeNet([10.0, 0.0, 0.0])
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[1] = 1
    mask[2] = 1
    obs = YuGiOhObservation(action_mask=mask)
    for seed in range(50):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        a = opp.select_action(obs)
        assert a in (1, 2), f"illegal action {a} was sampled at seed {seed}"
