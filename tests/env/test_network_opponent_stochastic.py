"""Tests for NetworkOpponent stochastic-sampling mode."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import torch.nn as nn

from yugioh_env.opponent import NetworkOpponent


class _FakeNet(nn.Module):
    """Minimal stand-in: ignores inputs, returns canned logits + dummy value + hx."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def init_hx(self, batch_size: int, device):
        return None

    def forward(self, cards, global_state, actions, mask, hx=None):
        return self._logits, torch.zeros(1, 1), hx


def _make_obs(n_actions: int) -> dict:
    return {
        "cards": np.zeros((1, 1), dtype=np.uint8),
        "global_state": np.zeros((1,), dtype=np.uint8),
        "actions": np.zeros((1, 1), dtype=np.uint8),
        "action_mask": np.ones((n_actions,), dtype=np.int8),
    }


def test_stochastic_false_is_argmax() -> None:
    net = _FakeNet([3.0, 1.0, 0.5])
    opp = NetworkOpponent(net, stochastic=False)
    opp.set_observation(_make_obs(3))
    assert opp.select_action({}, num_actions=3) == 0


def test_stochastic_true_samples_distribution() -> None:
    # Logits where action 2 is much more likely; with temperature 1 we
    # should see it dominate over many samples.
    net = _FakeNet([0.0, 0.0, 5.0])
    counts = [0, 0, 0]
    for seed in range(200):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        opp.set_observation(_make_obs(3))
        counts[opp.select_action({}, num_actions=3)] += 1
    assert counts[2] > counts[0]
    assert counts[2] > counts[1]
    assert counts[2] > 100  # vast majority


def test_stochastic_low_temperature_approaches_argmax() -> None:
    # Temperature very low (0.01) effectively == argmax.
    net = _FakeNet([3.0, 1.0, 0.5])
    torch.manual_seed(0)
    opp = NetworkOpponent(net, stochastic=True, temperature=0.01)
    opp.set_observation(_make_obs(3))
    assert opp.select_action({}, num_actions=3) == 0


def test_stochastic_temperature_zero_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=0.0)


def test_stochastic_negative_temperature_raises() -> None:
    net = _FakeNet([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        NetworkOpponent(net, stochastic=True, temperature=-0.5)


def test_stochastic_respects_action_mask() -> None:
    # Highest logit at action 0, but mask it out — sample must fall on 1 or 2.
    net = _FakeNet([10.0, 0.0, 0.0])
    obs = _make_obs(3)
    obs["action_mask"] = np.array([0, 1, 1], dtype=np.int8)
    for seed in range(50):
        torch.manual_seed(seed)
        opp = NetworkOpponent(net, stochastic=True, temperature=1.0)
        opp.set_observation(obs)
        a = opp.select_action({}, num_actions=3)
        assert a != 0, f"masked action 0 was sampled at seed {seed}"
