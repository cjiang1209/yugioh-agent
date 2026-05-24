"""Test GameState message-driven updates."""

from yugioh_core.constants import (
    LOCATION_DECK,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    MSG_DAMAGE,
    MSG_DRAW,
    MSG_LPUPDATE,
    MSG_MOVE,
    MSG_NEW_PHASE,
    MSG_NEW_TURN,
    MSG_PAY_LPCOST,
    MSG_RECOVER,
    MSG_START,
    MSG_WIN,
)
from yugioh_env.game_state import GameState


def test_initial_state():
    """GameState starts with default values."""
    gs = GameState()
    assert gs.lp == [8000, 8000]
    assert gs.deck_count == [0, 0]
    assert gs.hand_count == [0, 0]
    assert gs.turn_count == 0
    assert gs.is_finished is False


def test_msg_start_initializes_counts():
    """MSG_START should set deck_count and extra_count."""
    gs = GameState()
    gs.update(
        {
            "msg_type": MSG_START,
            "lp": [8000, 8000],
            "deck_count": [40, 40],
            "extra_count": [15, 15],
        }
    )
    assert gs.deck_count == [40, 40]
    assert gs.extra_count == [15, 15]


def test_draw_decrements_deck():
    """MSG_DRAW should move cards from deck to hand."""
    gs = GameState()
    gs.deck_count = [40, 40]
    gs.update(
        {
            "msg_type": MSG_DRAW,
            "player": 0,
            "cards": [{"code": 1}, {"code": 2}, {"code": 3}, {"code": 4}, {"code": 5}],
        }
    )
    assert gs.deck_count[0] == 35
    assert gs.hand_count[0] == 5
    # Player 1 unaffected
    assert gs.deck_count[1] == 40
    assert gs.hand_count[1] == 0


def test_draw_without_initialization_stays_zero():
    """If deck_count is never initialized, draws clamp to 0 (the bug scenario)."""
    gs = GameState()
    assert gs.deck_count == [0, 0]
    gs.update(
        {
            "msg_type": MSG_DRAW,
            "player": 0,
            "cards": [{"code": 1}, {"code": 2}],
        }
    )
    # deck_count stays at 0 due to max(0, ...) — this is wrong but won't go negative
    assert gs.deck_count[0] == 0
    assert gs.hand_count[0] == 2


def test_damage_reduces_lp():
    """MSG_DAMAGE reduces LP, clamped to 0."""
    gs = GameState()
    gs.update({"msg_type": MSG_DAMAGE, "player": 0, "amount": 3000})
    assert gs.lp[0] == 5000
    gs.update({"msg_type": MSG_DAMAGE, "player": 0, "amount": 9000})
    assert gs.lp[0] == 0


def test_recover_increases_lp():
    """MSG_RECOVER increases LP."""
    gs = GameState()
    gs.update({"msg_type": MSG_DAMAGE, "player": 1, "amount": 5000})
    gs.update({"msg_type": MSG_RECOVER, "player": 1, "amount": 2000})
    assert gs.lp[1] == 5000


def test_lpupdate_sets_lp():
    """MSG_LPUPDATE sets LP directly."""
    gs = GameState()
    gs.update({"msg_type": MSG_LPUPDATE, "player": 0, "lp": 1234})
    assert gs.lp[0] == 1234


def test_pay_lpcost():
    """MSG_PAY_LPCOST reduces LP."""
    gs = GameState()
    gs.update({"msg_type": MSG_PAY_LPCOST, "player": 1, "amount": 2000})
    assert gs.lp[1] == 6000


def test_new_turn_and_phase():
    """MSG_NEW_TURN and MSG_NEW_PHASE update turn/phase tracking."""
    gs = GameState()
    gs.update({"msg_type": MSG_NEW_TURN, "player": 0})
    assert gs.turn_count == 1
    assert gs.current_player == 0
    gs.update({"msg_type": MSG_NEW_PHASE, "phase": 0x08})
    assert gs.phase == 0x08
    gs.update({"msg_type": MSG_NEW_TURN, "player": 1})
    assert gs.turn_count == 2
    assert gs.current_player == 1


def test_win():
    """MSG_WIN sets finished state."""
    gs = GameState()
    gs.update({"msg_type": MSG_WIN, "player": 1})
    assert gs.is_finished is True
    assert gs.winner == 1


def test_move_updates_zone_counts():
    """MSG_MOVE should decrement source and increment destination."""
    gs = GameState()
    gs.deck_count = [40, 40]
    # Move a card from deck to hand (player 0)
    gs.update(
        {
            "msg_type": MSG_MOVE,
            "prev_controller": 0,
            "prev_location": LOCATION_DECK,
            "cur_controller": 0,
            "cur_location": LOCATION_HAND,
        }
    )
    assert gs.deck_count[0] == 39
    assert gs.hand_count[0] == 1

    # Move from hand to monster zone
    gs.update(
        {
            "msg_type": MSG_MOVE,
            "prev_controller": 0,
            "prev_location": LOCATION_HAND,
            "cur_controller": 0,
            "cur_location": LOCATION_MZONE,
        }
    )
    assert gs.hand_count[0] == 0
    assert gs.mzone_count[0] == 1

    # Move from monster zone to graveyard
    gs.update(
        {
            "msg_type": MSG_MOVE,
            "prev_controller": 0,
            "prev_location": LOCATION_MZONE,
            "cur_controller": 0,
            "cur_location": LOCATION_GRAVE,
        }
    )
    assert gs.mzone_count[0] == 0
    assert gs.grave_count[0] == 1


def test_reset():
    """Reset should restore all defaults."""
    gs = GameState()
    gs.lp = [100, 200]
    gs.deck_count = [10, 20]
    gs.turn_count = 5
    gs.is_finished = True
    gs.winner = 1
    gs.reset()
    assert gs.lp == [8000, 8000]
    assert gs.deck_count == [0, 0]
    assert gs.turn_count == 0
    assert gs.is_finished is False
    assert gs.winner == -1
