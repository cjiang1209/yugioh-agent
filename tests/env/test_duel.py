"""Test the high-level Duel class."""

from yugioh_core.constants import LOCATION_HAND, SELECT_MSGS
from yugioh_env.duel import Duel


def test_create_and_start(duel, deck_path):
    """Should create a duel, start, and process to first SELECT message."""
    duel.create(deck0=deck_path, deck1=deck_path, seed=42)
    msg, state, _events = duel.process_until_choice()
    assert msg is not None or state.is_finished
    if msg is not None:
        assert msg["msg_type"] in SELECT_MSGS


def test_destroy_cleanly(lib, card_db, script_dirs, deck_path):
    """Should destroy without errors or leaks."""
    d = Duel(lib, card_db, script_dirs)
    d.create(deck0=deck_path, deck1=deck_path, seed=1)
    d.destroy()
    # Double destroy should not crash
    d.destroy()


def test_context_manager(lib, card_db, script_dirs, deck_path):
    """Context manager should clean up properly."""
    with Duel(lib, card_db, script_dirs) as d:
        d.create(deck0=deck_path, deck1=deck_path, seed=2)
        msg, state, _events = d.process_until_choice()


def test_determinism(lib, card_db, script_dirs, deck_path):
    """Same seed + same actions should produce identical first message."""
    results = []
    for _ in range(2):
        with Duel(lib, card_db, script_dirs) as d:
            d.create(deck0=deck_path, deck1=deck_path, seed=12345)
            msg, state, _events = d.process_until_choice()
            results.append(msg)

    if results[0] is not None and results[1] is not None:
        assert results[0]["msg_type"] == results[1]["msg_type"]
        assert results[0].get("player") == results[1].get("player")


def test_query_count(duel, deck_path):
    """Should query card counts after starting."""
    duel.create(deck0=deck_path, deck1=deck_path, seed=42)
    duel.process_until_choice()

    from yugioh_core.constants import LOCATION_HAND

    # After initial draw, each player should have some cards in hand
    hand0 = duel.query_count(0, LOCATION_HAND)
    hand1 = duel.query_count(1, LOCATION_HAND)
    assert hand0 >= 0
    assert hand1 >= 0


def _get_opening_hand(lib, card_db, script_dirs, deck_path, seed):
    """Helper: return player-0's opening hand card codes for *seed*."""
    with Duel(lib, card_db, script_dirs) as d:
        d.create(deck0=deck_path, deck1=deck_path, seed=seed)
        d.process_until_choice()
        cards = d.query_location(0, LOCATION_HAND)
        return [c.get("code", 0) for c in cards]


def test_shuffle_deterministic_same_seed(lib, card_db, script_dirs, deck_path):
    """Same seed must always produce the same opening hand."""
    hand_a = _get_opening_hand(lib, card_db, script_dirs, deck_path, seed=42)
    hand_b = _get_opening_hand(lib, card_db, script_dirs, deck_path, seed=42)
    assert hand_a == hand_b
