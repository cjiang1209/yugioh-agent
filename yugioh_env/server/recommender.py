"""Web-layer loader + inference helper for the action recommender.

The recommender suggests moves for the *human* player and is configured
independently of the opponent via ``YUGIOH_RECOMMENDER``. It accepts the full
opponent spec grammar (``random`` / ``greedy`` / ``model:PATH`` /
``ygo-agent[:url]``) and is built with the same ``make_opponent`` factory.
"""

from __future__ import annotations

import os

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


def recommend_action_index(recommender: Opponent, obs: YuGiOhObservation) -> int:
    """Run the recommender on a live observation and return its chosen action index.

    Every recommender now takes the full observation directly (there is no
    longer a needs_observation split). The mask is dense
    (``mask[:num_actions] = 1``), so the returned index is a legal slot
    directly usable as an ``EngineAction.index``. The caller must ensure
    ``obs`` is non-terminal with at least one legal action.
    """
    action_index, _ = recommender.select_action(obs)
    return action_index
