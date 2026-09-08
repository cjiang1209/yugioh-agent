# tests/env/test_encoder_goldens.py
"""Guards the frozen observation capture itself.

The comparison of `encode_observation` against these goldens lives in
`tests/rl/test_obs_encoder_equivalence.py`. What is checked here is that the
capture is worth comparing against: a fixture that never reached a branch
protects nothing on that branch.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from core.constants import LOCATION_OVERLAY
from core.encoding import ACTION_LAYOUT, EVENT_LAYOUT

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def goldens():
    return np.load(FIXTURES / "encoder_goldens_observations.npz")


def _rows(goldens, suffix: str) -> list[np.ndarray]:
    arrays = [goldens[k] for k in goldens.files if k.endswith(suffix)]
    assert arrays, f"no {suffix} arrays captured"
    return arrays


def _fills(arrays, layout, field: str) -> bool:
    """Does any captured row carry a non-zero value in `field`?"""
    at, width = layout.span(field)
    return any(a[..., at : at + width].any() for a in arrays)


def test_goldens_cover_the_sparsely_filled_action_fields(goldens) -> None:
    """A field no captured action ever fills is a field the goldens do not pin.

    `subsequence` is the engine's overloaded loc_info slot: an overlay stack
    index when the action's location carries LOCATION_OVERLAY, a position
    bitmask otherwise. Only a pick from an Xyz monster's attached materials
    reaches the first meaning, and that needs an Xyz Summon to land under
    random play, so location is checked rather than the slot itself. The other
    two are merely rare: `position` is filled by an effect activation alone,
    `index` by the prompts that carry an engine index.
    """
    actions = _rows(goldens, "_actions")
    at = ACTION_LAYOUT.offsets["location"]
    assert any((a[:, at] & LOCATION_OVERLAY).any() for a in actions), (
        "no action carries LOCATION_OVERLAY -- no Xyz material pick was captured"
    )
    for field in ("position", "index"):
        assert _fills(actions, ACTION_LAYOUT, field), (
            f"every action golden has {field} == 0 -- nothing pins that field"
        )


def test_goldens_cover_the_chain_and_event_buffers(goldens) -> None:
    """Both buffers sit empty for most of a duel, so a capture holding only
    zeros would pin neither layout.

    The attack fields and `desc` come from event kinds the capture does reach.
    `hint_type` and `hint_value` do not: they carry a declaration hint, which
    only MSG_HINT produces, and random legal play never announces a race,
    attribute, card or number across the bundled decks. Those two fields are
    zero throughout the fixture, so nothing here pins them -- their layout
    rests on the full-width pack check and the event-encoder unit tests.
    """
    chain, events = _rows(goldens, "_chain"), _rows(goldens, "_events")
    assert any(c.any() for c in chain), "every pending_chain golden is empty"
    assert any(e.any() for e in events), "every event_history golden is empty"

    # A capture that only ever recorded one event kind would leave the fields
    # the other kinds fill sitting at zero.
    for field in ("target_code", "target_location", "target_sequence", "desc"):
        assert _fills(events, EVENT_LAYOUT, field), (
            f"every event golden has {field} == 0 -- the kind that fills it was never reached"
        )
