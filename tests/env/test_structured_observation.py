"""The structured fields must describe exactly the board the packed arrays
encode, and must carry everything packing needs. While both representations
exist, they can be compared row by row.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from yugioh_core.encoding import decode_u16, decode_u32, encode_card
from yugioh_env.models import CardState


def test_card_state_can_feed_encode_card() -> None:
    """Every encode_card parameter must be a CardState field, or packing needs
    a lookup the encoder does not have."""
    fields = {f.name for f in dataclasses.fields(CardState)}
    missing = set(inspect.signature(encode_card).parameters) - fields
    assert not missing, f"CardState cannot feed encode_card: missing {sorted(missing)}"


@pytest.fixture
def obs(lib, db_path, script_dirs, deck_path):
    """A freshly dealt duel's first observation."""
    from yugioh_env.deck_parser import parse_ydk
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    deck = parse_ydk(deck_path)
    try:
        yield env.reset(seed=7, deck0=deck, deck1=deck, agent_player=0)
    finally:
        env.close()


def test_card_states_match_the_packed_rows(obs) -> None:
    packed = np.asarray(obs.cards)
    assert obs.card_states, "no structured cards produced"
    for i, card in enumerate(obs.card_states):
        row = packed[i]
        assert decode_u32(row, 0) == card.code, f"row {i} code"
        assert int(row[4]) == card.location, f"row {i} location"
        assert int(row[5]) == card.sequence, f"row {i} sequence"
        assert int(row[7]) == card.controller, f"row {i} controller"
        assert decode_u16(row, 19) == card.attack, f"row {i} attack"


def test_global_matches_the_packed_scalars(obs) -> None:
    gs = np.asarray(obs.global_state)
    s = obs.global_
    assert s.my_lp == decode_u16(gs, 0)
    assert s.opp_lp == decode_u16(gs, 2)
    assert s.turn == int(gs[4])
    assert s.phase == decode_u16(gs, 5)
    assert s.is_my_turn == bool(gs[7])
    assert s.my_hand == int(gs[11])
    assert s.opp_hand == int(gs[16])


def test_hidden_cards_are_kept_in_place(obs) -> None:
    """Hidden cards occupy rows like any other card, so the structured list
    and the packed rows hold the same number of them.
    """
    packed = np.asarray(obs.cards)
    live = [r for r in packed if int(r[4]) != 0]
    assert len(obs.card_states) == len(live), "structured list dropped or added rows"
    hidden = [c for c in obs.card_states if c.code == 0]
    assert hidden, "no hidden cards in a freshly dealt duel -- check the fixture"
