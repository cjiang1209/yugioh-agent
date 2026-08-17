"""Web-layer loader + inference helper for the action recommender.

The recommender suggests moves for the *human* player and is configured
independently of the opponent via ``YUGIOH_RECOMMENDER``. It accepts the full
opponent spec grammar (``random`` / ``greedy`` / ``model:PATH`` /
``ygo-agent[:url]``) and is built with the same ``make_opponent`` factory.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from yugioh_env.models import YuGiOhObservation
from yugioh_env.opponent import Opponent, make_opponent


def make_recommender(
    spec: str | None, *, seed: int | None = None, device: str = "cpu"
) -> Opponent | None:
    """Build a recommender from a spec string, or ``None`` when disabled.

    Returns ``None`` when ``spec`` is falsy (feature off). Otherwise delegates
    to ``make_opponent`` (which raises ``ValueError`` on unknown/invalid specs).
    """
    if not spec:
        return None
    return make_opponent(spec, seed=seed, device=device)


def recommender_spec_from_env() -> str | None:
    """Read the recommender spec from ``YUGIOH_RECOMMENDER`` (None if unset/empty)."""
    return os.environ.get("YUGIOH_RECOMMENDER") or None


def recommender_device_from_env() -> str:
    """Read the recommender device from ``YUGIOH_RECOMMENDER_DEVICE`` (default cpu)."""
    return os.environ.get("YUGIOH_RECOMMENDER_DEVICE", "cpu")


@dataclass(frozen=True)
class Recommendation:
    """What the recommender produced for one prompt.

    ``action_index`` is always present. The two readouts are ``None`` for
    recommenders without a value head (``random`` / ``greedy`` / ``ygo-agent``),
    which pick an index but have nothing to inspect.
    """

    action_index: int
    value: float | None
    action_probs: list[float] | None

    def to_dict(self) -> dict:
        """The wire shape, field-for-field, like ``ActionDescriptor.to_dict``."""
        return asdict(self)


def recommend(recommender: Opponent, obs: YuGiOhObservation) -> Recommendation:
    """Run the recommender on a live observation and return its choice.

    Every recommender takes the full observation directly. The mask is dense
    (``mask[:num_actions] = 1``), so ``action_index`` is a legal slot directly
    usable as an ``EngineAction.index`` and ``action_probs`` -- when present --
    is index-aligned with the same action list. The caller must ensure ``obs``
    is non-terminal with at least one legal action.

    The readouts come from the forward pass that chose the action; reading them
    costs no additional inference.
    """
    action_index, inference = recommender.select_action(obs)
    if inference is None:
        return Recommendation(action_index=action_index, value=None, action_probs=None)
    return Recommendation(
        action_index=action_index,
        value=inference.value,
        action_probs=inference.action_probs,
    )
