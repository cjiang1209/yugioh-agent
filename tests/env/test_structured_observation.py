"""The structured fields must describe the board the engine reports, and must
carry everything packing needs.

The engine is the oracle for the board: `GameState` and `query_location` for
the values, and the interleaving rule for the ordering.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from yugioh_core.encoding import encode_card
from yugioh_env.models import CardState


def test_card_state_can_feed_encode_card() -> None:
    """Every encode_card parameter must be a CardState field, or packing needs
    a lookup the encoder does not have."""
    fields = {f.name for f in dataclasses.fields(CardState)}
    missing = set(inspect.signature(encode_card).parameters) - fields
    assert not missing, f"CardState cannot feed encode_card: missing {sorted(missing)}"


@pytest.fixture
def env(lib, db_path, script_dirs):
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    environment = YuGiOhEnvironment({})
    try:
        yield environment
    finally:
        environment.close()


@pytest.fixture
def obs(env, deck_path):
    """The opening observation of a freshly dealt duel."""
    from yugioh_env.deck_parser import parse_ydk

    deck = parse_ydk(deck_path)
    return env.reset(seed=7, deck0=deck, deck1=deck, agent_player=0)


def test_cards_carry_the_engine_coordinates(env, obs) -> None:
    """Every entry names a real zone, and the codes the engine reports for the
    agent's own hand all appear."""
    from yugioh_core.constants import LOCATION_HAND

    assert obs.cards, "no structured cards produced"
    # location == 0 is not a valid zone bitmask and renders as "deck".
    assert all(c.location != 0 for c in obs.cards)

    engine_hand = sorted(c["code"] for c in env.query_location(0, LOCATION_HAND))
    obs_hand = sorted(
        c.code for c in obs.cards if c.location == LOCATION_HAND and c.controller == 0
    )
    assert obs_hand == engine_hand


def test_global_matches_the_game_state(env, obs) -> None:
    engine = env._duel.game_state
    s = obs.global_state
    assert s.my_lp == engine.lp[0]
    assert s.opp_lp == engine.lp[1]
    assert s.turn == engine.turn_count
    assert s.phase == engine.phase
    assert s.is_my_turn == (engine.current_player == 0)
    assert s.my_hand == engine.hand_count[0]
    assert s.opp_hand == engine.hand_count[1]


def test_hidden_cards_are_kept_in_place(obs) -> None:
    """Hidden cards occupy a row like any other card: present, placed, and
    interleaved with the known ones.
    """
    hidden = [c for c in obs.cards if c.code == 0]
    assert hidden, "no hidden cards in a freshly dealt duel -- check the fixture"
    # A hidden card still names its zone and seat; only its identity is
    # withheld. Dropping them would shift every later row.
    assert all(c.location != 0 and not c.is_public for c in hidden)
    # They are interleaved, not appended: at least one known card follows
    # a hidden one.
    codes = [c.code for c in obs.cards]
    assert any(codes[i] == 0 and any(c != 0 for c in codes[i + 1 :]) for i in range(len(codes))), (
        "hidden cards all sort last -- the list was filtered or reordered"
    )
