"""Tests for cli/play_client.py:parse_global_state.

The global_state buffer is written by yugioh_env/observation.py. Its layout is
positional and `phase` is two bytes wide, so an off-by-one in any single field
silently shifts every field after it.
"""

from cli.play_client import parse_global_state

from yugioh_core.encoding import GLOBAL_FEATURES, encode_u16


def _global_state() -> list[int]:
    """A buffer laid out exactly as yugioh_env/observation.py writes it.

    Every integer field gets a DISTINCT value, so a shift cannot coincidentally
    agree: reading a neighbour always produces the wrong number. `phase` and the
    two life-point fields exceed one byte, which is what makes the widths matter.

    That trick does NOT work for the two booleans -- bool() collapses every
    non-zero neighbour to True -- so is_my_turn and is_finished each get a
    dedicated test below that sets them False against a truthy neighbour.
    """
    gs = [0] * GLOBAL_FEATURES
    gs[0], gs[1] = encode_u16(8000)  # my_lp
    gs[2], gs[3] = encode_u16(7000)  # opp_lp
    gs[4] = 5  # turn
    gs[5], gs[6] = encode_u16(0x100)  # phase — MAIN2, needs the high byte
    gs[7] = 1  # is_my_turn
    gs[8] = 2  # chain_count
    gs[9] = 11  # msg_type
    gs[10], gs[11], gs[12], gs[13], gs[14] = 30, 6, 3, 1, 9  # mine
    gs[15], gs[16], gs[17], gs[18], gs[19] = 28, 7, 4, 2, 8  # opponent
    gs[20] = 1  # is_finished
    return gs


def test_parse_global_state_matches_the_producer_layout():
    """Every field, so a one-slot shift anywhere in the buffer is caught."""
    assert parse_global_state(_global_state()) == {
        "my_lp": 8000,
        "opp_lp": 7000,
        "turn": 5,
        "phase": 0x100,
        "is_my_turn": True,
        "chain_count": 2,
        "msg_type": 11,
        "my_deck": 30,
        "my_hand": 6,
        "my_grave": 3,
        "my_banished": 1,
        "my_extra": 9,
        "opp_deck": 28,
        "opp_hand": 7,
        "opp_grave": 4,
        "opp_banished": 2,
        "opp_extra": 8,
        "is_finished": True,
    }


def test_parse_global_state_phase_keeps_its_high_byte():
    """MAIN2 (0x100) and END (0x200) live entirely in the high byte, so a
    one-byte phase read reports 0 for both and misreads is_my_turn from the
    byte it skipped."""
    gs = _global_state()
    gs[5], gs[6] = encode_u16(0x200)  # END
    gs[7] = 0  # opponent's turn
    parsed = parse_global_state(gs)
    assert parsed["phase"] == 0x200
    assert parsed["is_my_turn"] is False


def test_parse_global_state_is_finished_reads_its_own_slot():
    """is_finished is the last slot; reading one short lands on opp_extra and
    reports a live duel as over whenever the opponent holds extra-deck cards."""
    gs = _global_state()
    gs[19] = 9  # opponent has extra-deck cards
    gs[20] = 0  # but the duel is NOT finished
    assert parse_global_state(gs)["is_finished"] is False
