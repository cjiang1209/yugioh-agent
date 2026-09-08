"""The structured fields must describe the board the engine reports, and must
carry everything packing needs.

The engine is the oracle for the board: `GameState` and `query_location` for
the values, and the interleaving rule for the ordering.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.encoding import (
    ACTION_FEATURES,
    ACTION_LAYOUT,
    CARD_FEATURES,
    CARD_LAYOUT,
    CHAIN_ENTRY_FEATURES,
    CHAIN_LAYOUT,
    EVENT_ENTRY_FEATURES,
    EVENT_LAYOUT,
    GLOBAL_FEATURES,
    GLOBAL_LAYOUT,
    encode_action,
    encode_card,
    encode_chain_entry,
    encode_event_entry,
    encode_global,
)
from env.models import CardState, GlobalState, YuGiOhObservation

_ACTION_ARGS = [n for n, _ in ACTION_LAYOUT if not n.startswith("extra_idx")]


def _pack_action(**fields):
    """`encode_action` minus the extra_idx slots, which it has no parameters
    for -- callers here pass every field the layout names."""
    return encode_action(**{n: v for n, v in fields.items() if not n.startswith("extra_idx")})


def _assert_fields(layout, row, values) -> None:
    """Read every field back at the offset a reader would use.

    Each caller gives its fields distinct values, so any transposition in a
    packer sends one somewhere it is not expected. The flag fields are the
    limit of that: a packer writes each as 0 or 1, so a transposition among
    two flags changes no byte and no value table can catch it.
    """
    for name, _ in layout:
        got = layout.read(row, name)
        assert got == int(values[name]), f"{name} read {got}, expected {values[name]}"


def test_every_card_field_lands_at_its_own_offset() -> None:
    """Drive the real board loop with a distinct value per field and read each
    one back where the decoder looks for it.

    This is what pins the hand-written argument list in `_encode_cards`:
    `CardState` is keyword-only, so its declaration order governs nothing, and
    two fields transposed at that call site still encode. The frozen goldens
    catch such a swap only on a board where the two values happen to differ.

    All three flags are set here, leaving 0 free for `controller`, which would
    otherwise share a value with one of them. The goldens pin `is_public` and
    `negated`, which a real board varies; they cannot pin `is_overlay`, which
    has no producer and stays zero like the extra_idx slots.
    """
    from rl.obs_encoder import _encode_cards

    values = {
        "code": 0x01020304,
        "location": 7,
        "sequence": 11,
        "position": 13,
        "controller": 0,
        "is_public": True,
        "card_type": 0x05060708,
        "level": 17,
        "attribute": 19,
        "race": 0x090A0B0C,
        "attack": 1234,
        "defense": 5678,
        "lscale": 23,
        "rscale": 29,
        "link_marker": 4321,
        "counter_count": 31,
        "negated": True,
        "is_overlay": True,
    }
    assert set(values) == {f.name for f in dataclasses.fields(CardState)}
    row = _encode_cards(YuGiOhObservation(cards=[CardState(**values)]))[0]
    _assert_fields(CARD_LAYOUT, row, values)


def test_every_action_field_lands_at_its_own_offset() -> None:
    """The packer's counterpart of the card check: a distinct value per
    argument, read back at the offset the decoder uses.

    `direct_attackable` is the only flag here, so no other field may carry 1.
    """
    values = dict(
        zip(
            _ACTION_ARGS,
            (2, 3, 0x0A0B0C0D, 0, 5, 260, 6, 9, True, 12, 14, 16, 18, 20, 777),
            strict=True,
        )
    )
    buf = _pack_action(**values)
    # The extra_idx slots have no argument; the packer writes them zero.
    _assert_fields(ACTION_LAYOUT, buf, values | {"extra_idx_0": 0, "extra_idx_1": 0})


def test_every_global_field_lands_at_its_own_offset() -> None:
    """Drive the real encoder with a distinct value per field.

    `is_my_turn` and `is_finished` are the flags here, so no other field may
    carry 0 or 1.
    """
    from rl.obs_encoder import _encode_global

    values = {
        "my_lp": 8000,
        "opp_lp": 7000,
        "turn": 12,
        "phase": 0x0200,
        "is_my_turn": True,
        "chain_count": 3,
        "msg_type": 11,
        "my_deck": 17,
        "my_hand": 19,
        "my_grave": 23,
        "my_banished": 29,
        "my_extra": 31,
        "opp_deck": 37,
        "opp_hand": 41,
        "opp_grave": 43,
        "opp_banished": 47,
        "opp_extra": 53,
        "is_finished": False,
    }
    assert set(values) == {f.name for f in dataclasses.fields(GlobalState)}
    row = _encode_global(YuGiOhObservation(global_state=GlobalState(**values)))
    _assert_fields(GLOBAL_LAYOUT, row, values)


def test_every_field_accepts_its_full_width() -> None:
    """A clamp narrower than its field truncates a value the layout allows.

    The packers narrow by hand -- `min(sequence, 255)` beside `("sequence",
    "B")` -- so widening a field moves the offsets and leaves the clamp. Here
    each field is packed with the largest value its declared width holds and
    read back unchanged.

    The flag fields are excluded: a packer writes them as 0 or 1, so the
    widest value is not what comes back.
    """
    packers = [
        ("card", CARD_LAYOUT, encode_card, {"is_public", "negated", "is_overlay"}),
        ("action", ACTION_LAYOUT, _pack_action, {"direct_attackable"}),
        ("global", GLOBAL_LAYOUT, encode_global, {"is_my_turn", "is_finished"}),
        ("chain", CHAIN_LAYOUT, encode_chain_entry, set()),
        ("event", EVENT_LAYOUT, encode_event_entry, set()),
    ]
    for label, layout, pack, flags in packers:
        for name, code in layout:
            if name in flags or name.startswith("extra_idx"):
                continue
            widest = (1 << (8 * layout.span(name)[1])) - 1
            row = pack(**{n: widest if n == name else 0 for n, _ in layout})
            got = layout.read(row, name)
            assert got == widest, (
                f"{label}.{name} is declared {code!r} but came back {got}, "
                f"not {widest} -- the packer's clamp is narrower than the field"
            )


def test_layouts_fit_their_rows() -> None:
    """A struct wider than its row would run into the next one.

    Rows are packed into a shared buffer at `i * FEATURES`, so an overrun
    corrupts the following row's leading bytes and only raises on the last row
    of a full block -- silently, for any shorter block.

    Only the card row carries padding. The other four are exact, and are
    asserted exact: a dropped field would leave dead bytes in a width the
    checkpoint and the buffers already depend on.
    """
    assert CARD_LAYOUT.struct.size <= CARD_FEATURES
    assert ACTION_LAYOUT.struct.size == ACTION_FEATURES
    assert GLOBAL_LAYOUT.struct.size == GLOBAL_FEATURES
    assert CHAIN_LAYOUT.struct.size == CHAIN_ENTRY_FEATURES
    assert EVENT_LAYOUT.struct.size == EVENT_ENTRY_FEATURES


@pytest.fixture
def env(lib, db_path, script_dirs):
    from env.server.environment import YuGiOhEnvironment

    environment = YuGiOhEnvironment({})
    try:
        yield environment
    finally:
        environment.close()


@pytest.fixture
def obs(env, deck_path):
    """The opening observation of a freshly dealt duel."""
    from env.deck_parser import parse_ydk

    deck = parse_ydk(deck_path)
    return env.reset(seed=7, deck0=deck, deck1=deck, agent_player=0)


def test_cards_carry_the_engine_coordinates(env, obs) -> None:
    """Every entry names a real zone, and the codes the engine reports for the
    agent's own hand all appear."""
    from core.constants import LOCATION_HAND

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
