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

from yugioh_core.constants import LOCATION_OVERLAY

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_goldens_cover_every_position_shaped_route() -> None:
    """The overlay branch is unreachable on most decks, so a capture that never
    hits it protects nothing.

    `a[:, 7] & LOCATION_OVERLAY` finds the rows the overlay branch applies
    to: an action whose location carries LOCATION_OVERLAY is one where byte
    10 holds an overlay stack index instead of a position bitmask. Byte 7 is
    the discriminator; byte 10 is the field it changes the meaning of."""
    data = np.load(FIXTURES / "encoder_goldens_observations.npz")
    actions = [data[k] for k in data.files if k.endswith("_actions")]
    assert actions, "no observations captured"
    assert sum(int(((a[:, 7] & LOCATION_OVERLAY) != 0).sum()) for a in actions) > 0, (
        "byte-10 overlay missing"
    )
    assert sum(int((a[:, 11] != 0).sum()) for a in actions) > 0, "byte-11 chain route missing"
    assert sum(int((a[:, 16] != 0).sum()) for a in actions) > 0, "byte-16 index route missing"
