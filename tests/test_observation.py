"""Test observation encoding."""

import numpy as np
import pytest

from yugioh_env.observation import (
    MAX_CARDS,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    build_observation,
    encode_card,
)
from yugioh_env.game_state import GameState


def test_encode_card_shape():
    """Card encoding should produce correct shape."""
    feat = encode_card(
        code=89631139, location=0x02, sequence=0, position=0x1,
        controller=0, is_public=True, card_type=0x11,
        level=8, attribute=0x10, race=0x2000, attack=3000, defense=2500,
    )
    assert feat.shape == (CARD_FEATURES,)
    assert feat.dtype == np.uint8


def test_encode_card_hidden():
    """Hidden card should have zero features except location/controller."""
    feat = encode_card(
        code=0, location=0x02, sequence=0, position=0,
        controller=1, is_public=False,
    )
    assert feat.shape == (CARD_FEATURES,)
    # code should be 0
    assert feat[0] == 0
    assert feat[1] == 0
    # controller should be 1
    assert feat[5] == 1
    # is_public should be 0
    assert feat[6] == 0


def test_build_observation_shapes():
    """Observation should have correct shapes."""
    gs = GameState()
    gs.lp = [8000, 7500]
    gs.turn_count = 1
    gs.phase = 0x04

    obs = build_observation(gs, None, agent_player=0)
    assert obs["cards"].shape == (MAX_CARDS, CARD_FEATURES)
    assert obs["global_state"].shape == (GLOBAL_FEATURES,)


def test_global_state_lp():
    """Global state should encode LP correctly."""
    gs = GameState()
    gs.lp = [8000, 4000]

    obs = build_observation(gs, None, agent_player=0)
    g = obs["global_state"]
    # my_lp = 8000 = 0x1F40 (cast to int to avoid numpy uint8 overflow)
    my_lp = int(g[0]) | (int(g[1]) << 8)
    assert my_lp == 8000
    # opp_lp = 4000 = 0x0FA0
    opp_lp = int(g[2]) | (int(g[3]) << 8)
    assert opp_lp == 4000
