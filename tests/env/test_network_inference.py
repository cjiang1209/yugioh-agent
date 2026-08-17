"""NetworkOpponent reports the readouts its forward pass already produces:
the value head scalar and the policy distribution."""

import pytest

torch = pytest.importorskip("torch")

from tests.env.conftest import obs_from_mask
from yugioh_core.encoding import MAX_ACTIONS  # noqa: E402
from yugioh_env.opponent import Inference, NetworkOpponent  # noqa: E402


class FakeNet:
    """Returns fixed logits and a fixed value, ignoring its inputs.

    Logits favour slot 1. The value is distinctive so a test can tell it apart
    from a probability or a default.
    """

    def __init__(self, value=0.375):
        self._value = value

    def init_hx(self, batch_size, device):
        return None

    def __call__(self, *, hx=None, **inputs):
        logits = torch.full((1, MAX_ACTIONS), -1.0)
        logits[0, 1] = 3.0
        values = torch.tensor([self._value])
        return logits, values, hx


def test_reports_value_and_probabilities():
    opp = NetworkOpponent(FakeNet(value=0.375))
    action, inference = opp.select_action(obs_from_mask(num_legal=3))

    assert action == 1
    assert isinstance(inference, Inference)
    assert inference.value == pytest.approx(0.375)
    # One entry per legal action, not per MAX_ACTIONS slot.
    assert len(inference.action_probs) == 3
    assert sum(inference.action_probs) == pytest.approx(1.0)
    # Slot 1 carries the highest logit, so it carries the highest probability.
    assert inference.action_probs[1] == pytest.approx(max(inference.action_probs))


def test_reported_probabilities_are_the_ones_sampled_from():
    """`action_probs` is the distribution the action came from, so a temperature
    that flattens the sampling flattens the report with it. Reporting a rescaled
    distribution instead would cost a second softmax that only the sampler's
    caller could tell apart -- and none of them read it."""
    greedy = NetworkOpponent(FakeNet())
    hot = NetworkOpponent(FakeNet(), stochastic=True, temperature=5.0)

    _, from_greedy = greedy.select_action(obs_from_mask())
    _, from_hot = hot.select_action(obs_from_mask())

    assert sum(from_hot.action_probs) == pytest.approx(1.0)
    # T > 1 pulls mass off the favoured slot, toward uniform.
    assert max(from_hot.action_probs) < max(from_greedy.action_probs)


def test_temperature_one_reports_the_plain_policy():
    """At T=1 the tempered and plain distributions coincide, so the greedy
    default reports the policy distribution itself. A stochastic opponent at
    T=1 must agree with it."""
    greedy = NetworkOpponent(FakeNet())
    neutral = NetworkOpponent(FakeNet(), stochastic=True, temperature=1.0)

    _, from_greedy = greedy.select_action(obs_from_mask())
    _, from_neutral = neutral.select_action(obs_from_mask())

    assert from_neutral.action_probs == pytest.approx(from_greedy.action_probs)


def test_illegal_slots_are_excluded_not_zeroed():
    """action_probs is index-aligned with the caller's actions[] list, which
    only ever holds the legal slots."""
    opp = NetworkOpponent(FakeNet())
    _, inference = opp.select_action(obs_from_mask(num_legal=2))
    assert len(inference.action_probs) == 2
