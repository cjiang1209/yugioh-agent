# tests/env/test_encoder_goldens.py
"""The frozen encoder output is the oracle for byte-identity.

These pass trivially today -- they compare the current encoder against a
capture of itself. Their purpose is to fail the moment a replacement encoder
diverges, so the capture has to exist before that encoder is written.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from yugioh_core.constants import LOCATION_OVERLAY
from yugioh_env.action_space import ActionMapper

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
ACTION_GOLDENS = json.loads((FIXTURES / "encoder_goldens_actions.json").read_text())


@pytest.mark.parametrize("msg_type_key", sorted(ACTION_GOLDENS))
def test_action_encoder_matches_golden(msg_type_key: str) -> None:
    golden = ACTION_GOLDENS[msg_type_key]
    mapper = ActionMapper()
    mapper.update({**golden["msg"], "msg_type": int(msg_type_key), "_agent_player": 0})
    assert mapper.get_action_features().tolist() == golden["actions"]
    assert mapper.get_action_mask().tolist() == golden["mask"]
    assert mapper.num_actions == golden["num_actions"]


def test_goldens_cover_every_position_shaped_route() -> None:
    """Guards the fixture itself. The overlay branch is unreachable on most
    decks, so a capture that never hits it protects nothing.

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
