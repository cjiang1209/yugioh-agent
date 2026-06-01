"""Test action space mapping."""

import struct

import numpy as np
import pytest

from yugioh_core.constants import (
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_RACE,
    MSG_ROCK_PAPER_SCISSORS,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_COUNTER,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    MSG_SORT_CARD,
    MSG_SORT_CHAIN,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_core.encoding import decode_u16, decode_u32
from yugioh_env import response_builder as rb
from yugioh_env.action_space import _ACTION_EXTRACTORS, ACTION_FEATURES, MAX_ACTIONS, ActionMapper


def test_yesno_actions():
    """Yes/No should produce exactly 2 actions."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    assert mapper.num_actions == 2

    mask = mapper.get_action_mask()
    assert mask.shape == (MAX_ACTIONS,)
    assert mask[0] == 1
    assert mask[1] == 1
    assert mask[2] == 0


def test_yesno_responses():
    """Yes/No responses should be valid 4-byte uint32."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})

    yes_resp = mapper.action_to_response(0)
    no_resp = mapper.action_to_response(1)
    assert len(yes_resp) == 4
    assert len(no_resp) == 4


def test_chain_no_forced():
    """Non-forced chain should include 'no chain' option."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "spe_count": 0,
            "forced": 0,
            "hint_timing": 0,
            "other_timing": 0,
            "chains": [
                {
                    "code": 100,
                    "controller": 0,
                    "location": 2,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
        }
    )
    # 1 chain + 1 "no chain" = 2 actions
    assert mapper.num_actions == 2


def test_position_actions():
    """Position selection should list available positions."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_POSITION,
            "player": 0,
            "code": 12345,
            "positions": POS_FACEUP_ATTACK | POS_FACEUP_DEFENSE,
        }
    )
    assert mapper.num_actions == 2


def test_option_actions():
    """Option selection should produce correct number of actions."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_OPTION,
            "player": 0,
            "options": [100, 200, 300],
        }
    )
    assert mapper.num_actions == 3


def test_action_features_shape():
    """Action features should have correct shape."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    features = mapper.get_action_features()
    assert features.shape == (MAX_ACTIONS, ACTION_FEATURES)
    assert features.dtype == np.uint8


def test_action_features_card_code_encoding():
    """Card code should be encoded as 4-byte uint32 LE in feat[2:6]."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 1,
            "cards": [
                {"code": 89631139, "controller": 0, "location": 2, "sequence": 3, "subsequence": 0},
            ],
        }
    )
    features = mapper.get_action_features()
    feat = features[0]
    # feat[0] = msg_type
    assert feat[0] == MSG_SELECT_CARD
    # feat[2:6] = code as uint32 LE (89631139 = 0x0557B1A3)
    code = decode_u32(feat, 2)
    assert code == 89631139
    # New 28-byte layout: [6]=controller, [7]=location, [8:10]=sequence (u16 LE), [16]=index
    assert feat[7] == 2  # location
    seq = decode_u16(feat, 8)
    assert seq == 3  # sequence
    assert feat[16] == 0  # index


def test_invalid_action_index():
    """Invalid action index should raise ValueError."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    with pytest.raises(ValueError):
        mapper.action_to_response(5)


# --- Place action tests (field_mask is from selecting player's perspective) ---


def test_place_actions_player0():
    """Player 0 selecting: bits 0-15 = player 0's zones."""
    mapper = ActionMapper()
    # Only SZONE seq 0 available for player 0 (bit 8 = 0, rest = 1)
    mask = 0xFFFFFFFF ^ (1 << 8)
    mapper.update({"msg_type": MSG_SELECT_PLACE, "player": 0, "count": 1, "field_mask": mask})
    assert mapper.num_actions == 1
    resp = mapper.action_to_response(0)
    player, loc, seq = struct.unpack("<BBB", resp)
    assert player == 0
    assert loc == LOCATION_SZONE
    assert seq == 0


def test_place_actions_player1():
    """Player 1 selecting: bits 0-15 = player 1's own zones (absolute player 1)."""
    mapper = ActionMapper()
    # Only SZONE seq 2 available for player 1 (bit 10 = 0, rest = 1)
    mask = 0xFFFFFFFF ^ (1 << 10)
    mapper.update({"msg_type": MSG_SELECT_PLACE, "player": 1, "count": 1, "field_mask": mask})
    assert mapper.num_actions == 1
    resp = mapper.action_to_response(0)
    player, loc, seq = struct.unpack("<BBB", resp)
    assert player == 1
    assert loc == LOCATION_SZONE
    assert seq == 2


def test_place_actions_opponent_zones():
    """Player 1 selecting from opponent (player 0) zones: bits 16-31."""
    mapper = ActionMapper()
    # Only MZONE seq 0 of opponent available (bit 16 = 0, rest = 1)
    mask = 0xFFFFFFFF ^ (1 << 16)
    mapper.update({"msg_type": MSG_SELECT_PLACE, "player": 1, "count": 1, "field_mask": mask})
    assert mapper.num_actions == 1
    resp = mapper.action_to_response(0)
    player, loc, seq = struct.unpack("<BBB", resp)
    assert player == 0  # opponent of player 1
    assert loc == LOCATION_MZONE
    assert seq == 0


# --- Card selection response format tests ---


def test_select_card_response_format():
    """Card response must have type(int32=0) + count(uint32) + uint32 indices."""
    resp = rb.build_select_card_response([3])
    assert len(resp) == 12  # 4 + 4 + 4
    typ, count, idx = struct.unpack("<iII", resp)
    assert typ == 0
    assert count == 1
    assert idx == 3


def test_select_card_response_multi():
    """Multi-card selection response."""
    resp = rb.build_select_card_response([0, 2, 5])
    assert len(resp) == 20  # 4 + 4 + 3*4
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 3
    indices = [struct.unpack_from("<I", resp, 8 + i * 4)[0] for i in range(3)]
    assert indices == [0, 2, 5]


def test_select_sum_response_format():
    """Sum response uses same type-discriminated format as card selection."""
    resp = rb.build_select_sum_response([1, 4])
    assert len(resp) == 16  # 4 + 4 + 2*4
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 2


# ─── MSG_SELECT_SUM multi-step tests ────────────────────────────────────────


def _sum_msg_multi(
    target_sum, cards, *, min_sel=None, max_sel=None, select_type=0, selected=None, must_cards=None
):
    """Build a MSG_SELECT_SUM message with multiple optional cards."""
    if select_type == 1:
        # Engine always sends min=0, max=0 for SelectWithSumGreater
        min_sel = 0
        max_sel = 0
    else:
        if min_sel is None:
            min_sel = 1
        if max_sel is None:
            max_sel = len(cards)
    msg = {
        "msg_type": MSG_SELECT_SUM,
        "player": 0,
        "_agent_player": 0,
        "select_type": select_type,
        "target_sum": target_sum,
        "min": min_sel,
        "max": max_sel,
        "must_cards": must_cards or [],
        "optional_cards": [
            {"code": 100 + i, "controller": 0, "location": 0x04, "sequence": i, "param": p}
            for i, p in enumerate(cards)
        ],
    }
    if selected is not None:
        msg["_selected"] = selected
    return msg


def test_select_sum_single_card_exact_match():
    """A card whose param equals target_sum completes in one pick."""
    mapper = ActionMapper()
    mapper.update(_sum_msg_multi(5, [5, 3, 2]))
    assert mapper.num_actions > 0
    # Card 0 (param=5) should complete immediately
    resp = mapper.action_to_response(0)
    assert resp is not None
    _, count = struct.unpack_from("<iI", resp, 0)
    assert count == 1


def test_select_sum_single_card_no_match_is_intermediate():
    """A card whose param != target_sum is intermediate (None), not terminal."""
    mapper = ActionMapper()
    mapper.update(_sum_msg_multi(5, [3, 2]))
    # Card 0 (param=3) alone doesn't sum to 5, but 3+2 does
    resp = mapper.action_to_response(0)
    assert resp is None


def test_select_sum_exact_two_step_completion():
    """Picking two cards across steps produces a valid response."""
    mapper = ActionMapper()
    # target=5, cards: [3, 2, 4]
    msg = _sum_msg_multi(5, [3, 2, 4])
    mapper.update(msg)

    # Step 1: pick card 0 (param=3) — intermediate
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: re-present with card 0 selected
    mapper.update(_sum_msg_multi(5, [3, 2, 4], selected=[0]))
    # Card 1 (param=2) → 3+2=5 → completes
    resp = mapper.action_to_response(0)  # first available is card 1
    assert resp is not None
    _, count = struct.unpack_from("<iI", resp, 0)
    assert count == 2


def test_select_sum_dead_end_filtered():
    """Cards that cannot participate in any valid completion are not offered."""
    mapper = ActionMapper()
    # target=5, max=2, cards: [3, 2, 4]
    # Valid combos: [3,2]=5. Card 4 can't pair with anything to make 5.
    mapper.update(_sum_msg_multi(5, [3, 2, 4], max_sel=2))
    codes = [a["code"] for a in mapper.actions if a.get("category") == 0]
    # Card 2 (code=102, param=4) should be filtered: 4+3=7≠5, 4+2=6≠5
    assert 102 not in codes
    # Cards 0 and 1 should be present
    assert 100 in codes
    assert 101 in codes


def test_select_sum_no_response_sent_for_impossible_sum():
    """No action produces a response when the selected cards don't sum correctly."""
    mapper = ActionMapper()
    # target=10, cards: [3, 2, 4] — no subset sums to 10
    mapper.update(_sum_msg_multi(10, [3, 2, 4]))
    # All cards should be filtered (no valid completion exists)
    assert mapper.num_actions == 0


def test_select_sum_atleast_two_step_completion():
    """select_type=1 uses range-based validation instead of exact sum."""
    mapper = ActionMapper()
    # target=5, cards: [3, 4]. select_type=1 (at-least).
    # Card 0 (param=3): max=3 < 5 → alone can't reach target → filtered
    # Card 1 (param=4): max=4 < 5 → alone can't reach target → filtered
    # But [3,4]: max=7 >= 5, min=7, smallest=3, min-smallest=4 < 5 → valid
    mapper.update(_sum_msg_multi(5, [3, 4], select_type=1))
    # Both cards should be offered (they form a valid pair)
    assert mapper.num_actions == 2
    # Neither alone completes (both intermediate)
    assert mapper.action_to_response(0) is None
    assert mapper.action_to_response(1) is None
    # After picking card 0, card 1 completes
    mapper.update(_sum_msg_multi(5, [3, 4], select_type=1, selected=[0]))
    resp = mapper.action_to_response(0)
    assert resp is not None


def test_select_sum_must_cards_included_in_sum():
    """must_cards' params are included in the sum check."""
    mapper = ActionMapper()
    # target=7, must_cards=[param=4], optional: [3, 2, 5]
    # must(4) + card 0(3) = 7 ✓, must(4) + card 1(2) = 6 ✗, must(4) + card 2(5) = 9 ✗
    must = [{"code": 999, "controller": 0, "location": 0x04, "sequence": 99, "param": 4}]
    mapper.update(_sum_msg_multi(7, [3, 2, 5], min_sel=1, max_sel=1, must_cards=must))
    # Only card 0 (param=3) should be offered (4+3=7)
    assert mapper.num_actions == 1
    resp = mapper.action_to_response(0)
    assert resp is not None
    assert mapper.actions[0]["code"] == 100  # card 0


def test_select_card_actions_response():
    """Full round-trip: MSG_SELECT_CARD -> ActionMapper -> response bytes."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 1,
            "cards": [
                {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
                {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
            ],
        }
    )
    assert mapper.num_actions == 2
    resp = mapper.action_to_response(1)
    typ, count, idx = struct.unpack("<iII", resp)
    assert typ == 0
    assert count == 1
    assert idx == 1  # selected second card


# --- MSG_SELECT_UNSELECT_CARD response format ---


def test_unselect_card_response_select():
    """Unselect card response sends returns[0]=1 and returns[1]=index."""
    resp = rb.build_select_unselect_card_response(2)
    assert len(resp) == 8
    val0, val1 = struct.unpack("<iI", resp)
    assert val0 == 1
    assert val1 == 2


def test_unselect_card_response_finish():
    """Unselect card finish sends returns[0]=-1."""
    resp = rb.build_select_unselect_card_response(-1)
    assert len(resp) == 4
    val = struct.unpack("<i", resp)[0]
    assert val == -1


def test_unselect_card_actions():
    """MSG_SELECT_UNSELECT_CARD actions produce correct response format."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_UNSELECT_CARD,
            "player": 0,
            "finishable": 1,
            "cancelable": 0,
            "min": 1,
            "max": 3,
            "selectable": [
                {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
                {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
                {"code": 300, "controller": 0, "location": 2, "sequence": 2, "subsequence": 0},
            ],
            "unselectable": [],
        }
    )
    # 3 selectable + 1 finish = 4 actions
    assert mapper.num_actions == 4

    # Selecting card at index 2
    resp = mapper.action_to_response(2)
    val0, val1 = struct.unpack("<iI", resp)
    assert val0 == 1
    assert val1 == 2

    # Finish action (last)
    resp = mapper.action_to_response(3)
    val = struct.unpack("<i", resp)[0]
    assert val == -1


# --- Multi-select card tests ---


def test_select_card_multi_select():
    """MSG_SELECT_CARD with min=2, max=2 uses two-step selection."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
            {"code": 300, "controller": 0, "location": 2, "sequence": 2, "subsequence": 0},
        ],
    }
    mapper.update(msg)
    # Step 1: 3 individual cards (no finish since min=max=2)
    assert mapper.num_actions == 3

    # Pick card 0 — not done yet (1 < max=2), returns None
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: caller injects _selected and re-calls update
    mapper.update({**msg, "_selected": [0]})
    assert mapper.num_actions == 2  # 2 remaining cards (no finish since min=max)

    # Pick card 2 — hits max=2, auto-completes
    resp = mapper.action_to_response(1)  # remaining card at index 1 = original card 2
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 2


# --- Tribute tests ---


def test_tribute_two_tributes():
    """MSG_SELECT_TRIBUTE with min=2, max=2 uses two-step selection."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "release_param": 1},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
            {"code": 300, "controller": 0, "location": 4, "sequence": 2, "release_param": 1},
        ],
    }
    mapper.update(msg)
    # Step 1: 3 individual cards, all return None (len=1 < max=2)
    assert mapper.num_actions == 3
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: pick card 0, inject _selected
    mapper.update({**msg, "_selected": [0]})
    # 2 remaining cards, both complete (total=2 >= min=2 and len=2 >= max=2)
    assert mapper.num_actions == 2
    resp = mapper.action_to_response(0)
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 2


def test_select_card_min_max_multi_step():
    """MSG_SELECT_CARD with min=1, max=2 uses multi-step selection."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
            {"code": 300, "controller": 0, "location": 2, "sequence": 2, "subsequence": 0},
            {"code": 400, "controller": 0, "location": 2, "sequence": 3, "subsequence": 0},
        ],
    }
    mapper.update(msg)
    # Step 1: 4 individual cards, no finish yet (0 < min=1)
    assert mapper.num_actions == 4

    features = mapper.get_action_features()
    # Each card action has num_selected = 1 (will be 1 after pick)
    for i in range(4):
        assert features[i][17] == 1
        assert features[i][1] == 0  # category = 0 (card pick)

    # Pick card 0 — returns None (multi-step in progress)
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: caller injects _selected
    mapper.update({**msg, "_selected": [0]})
    # 3 remaining cards + 1 finish option (1 >= min=1)
    assert mapper.num_actions == 4  # 3 cards + 1 finish
    features2 = mapper.get_action_features()
    # Last action is "finish" (category=1)
    assert features2[3][1] == 1  # category = 1
    assert features2[3][17] == 1  # num_selected = 1 (accumulated so far)
    # Card actions have num_selected = 2 (will be 2 after this pick)
    for i in range(3):
        assert features2[i][17] == 2
        assert features2[i][1] == 0  # category = 0

    # Pick card 2 — hits max=2, returns final response
    resp = mapper.action_to_response(1)  # index 1 in remaining = card 2 (original index)
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 2


def test_select_card_min_max_finish_early():
    """Multi-step: picking 'finish' after min selections."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 3,
        "cards": [
            {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
            {"code": 300, "controller": 0, "location": 2, "sequence": 2, "subsequence": 0},
        ],
    }
    mapper.update(msg)
    # Step 1: 3 cards, no finish
    assert mapper.num_actions == 3

    # Pick card 1
    resp = mapper.action_to_response(1)
    assert resp is None

    # Step 2: caller injects _selected
    mapper.update({**msg, "_selected": [1]})
    # 2 remaining cards + finish (1 >= min=1)
    assert mapper.num_actions == 3  # 2 cards + 1 finish

    # Pick "finish" (last action)
    resp = mapper.action_to_response(2)
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 1  # Only 1 card selected
    idx = struct.unpack_from("<I", resp, 8)[0]
    assert idx == 1  # Original card index 1


def test_tribute_double_release():
    """A card with release_param=2 offers finish after first pick."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "release_param": 2},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
        ],
    }
    mapper.update(msg)
    # Step 1: both cards return None (len=1 < max=2)
    assert mapper.num_actions == 2
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: after picking card 0 (release_param=2), can_finish fires
    mapper.update({**msg, "_selected": [0]})
    # 1 remaining card + 1 finish (total release=2 >= min=2)
    assert mapper.num_actions == 2
    # Card 1 completes (total=3 >= 2 and len=2 >= 2)
    resp = mapper.action_to_response(0)
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 2
    # Finish action sends just card 0
    resp_finish = mapper.action_to_response(1)
    assert resp_finish is not None
    typ, count = struct.unpack_from("<iI", resp_finish, 0)
    assert typ == 0
    assert count == 1


# --- MSG_SELECT_COUNTER response format ---


def test_select_counter_response_no_length_prefix():
    """Counter response is just int16 values — no length prefix."""
    resp = rb.build_select_counter_response([3, 0, 2])
    # Should be exactly 3 * 2 = 6 bytes (no length prefix)
    assert len(resp) == 6
    c0, c1, c2 = struct.unpack("<HHH", resp)
    assert c0 == 3
    assert c1 == 0
    assert c2 == 2


def test_select_counter_response_single():
    """Counter response for single card."""
    resp = rb.build_select_counter_response([5])
    assert len(resp) == 2
    val = struct.unpack("<H", resp)[0]
    assert val == 5


def test_select_counter_actions_response():
    """Full round-trip: MSG_SELECT_COUNTER -> ActionMapper -> response bytes."""
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_COUNTER,
            "player": 0,
            "counter_type": 0x01,
            "count": 3,
            "cards": [
                {"code": 100, "controller": 0, "location": 4, "sequence": 0, "counter_count": 2},
                {"code": 200, "controller": 0, "location": 4, "sequence": 1, "counter_count": 5},
            ],
        }
    )
    assert mapper.num_actions >= 1
    resp = mapper.action_to_response(0)
    # Response should be raw int16 values, no length prefix
    assert len(resp) == 4  # 2 cards * 2 bytes each
    c0, c1 = struct.unpack("<HH", resp)
    # First action: remove min(counter_count, count) from card 0
    assert c0 + c1 > 0


# --- Tribute multi-step tests ---


def test_tribute_multi_step_with_finish():
    """Tribute with min=2, max=3: finish offered after 2 cards, completes at 3."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 3,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "release_param": 1},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
            {"code": 300, "controller": 0, "location": 4, "sequence": 2, "release_param": 1},
        ],
    }
    mapper.update(msg)
    # Step 1: 3 cards, no finish (total=0 < min=2)
    assert mapper.num_actions == 3

    # Pick card 0
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: 2 remaining, no finish yet (total=1 < min=2)
    mapper.update({**msg, "_selected": [0]})
    assert mapper.num_actions == 2

    # Pick card 1
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 3: 1 remaining card + finish (total=2 >= min=2)
    mapper.update({**msg, "_selected": [0, 1]})
    assert mapper.num_actions == 2  # 1 card + 1 finish

    # Picking the last card hits max=3 and completes
    resp = mapper.action_to_response(0)
    assert resp is not None
    typ, count = struct.unpack_from("<iI", resp, 0)
    assert typ == 0
    assert count == 3

    # Or pick finish with 2 cards
    resp_finish = mapper.action_to_response(1)
    assert resp_finish is not None
    typ, count = struct.unpack_from("<iI", resp_finish, 0)
    assert typ == 0
    assert count == 2


def test_tribute_release_param_finish_early():
    """release_param=2 card allows finish after 1 pick via can_finish."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "release_param": 2},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
        ],
    }
    mapper.update(msg)
    # Step 1: pick release_param=2 card — returns None (len=1 < max=2)
    resp = mapper.action_to_response(0)
    assert resp is None

    # Step 2: can_finish fires (total=2 >= min=2)
    mapper.update({**msg, "_selected": [0]})
    # 1 remaining card + 1 finish = 2 actions
    assert mapper.num_actions == 2

    # Finish sends response with 1 card
    resp_finish = mapper.action_to_response(1)
    assert resp_finish is not None
    typ, count = struct.unpack_from("<iI", resp_finish, 0)
    assert typ == 0
    assert count == 1
    idx = struct.unpack_from("<I", resp_finish, 8)[0]
    assert idx == 0  # original card index 0


def test_tribute_num_selected_feature():
    """Feature byte 9 (num_selected) reflects multi-step accumulation."""
    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "release_param": 1},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
            {"code": 300, "controller": 0, "location": 4, "sequence": 2, "release_param": 1},
        ],
    }
    mapper.update(msg)
    features = mapper.get_action_features()
    # Step 1: each card action has num_selected = 1 (will be 1 after pick)
    for i in range(3):
        assert features[i][17] == 1

    # Step 2: after picking card 0
    mapper.update({**msg, "_selected": [0]})
    features2 = mapper.get_action_features()
    # Remaining card actions have num_selected = 2 (will be 2 after pick)
    for i in range(2):
        assert features2[i][17] == 2


# --- Meta field boundary-case tests (Tasks 3-9) ---


def test_announce_race_meta_unknown_race_falls_back_to_hex():
    """An unmapped race bit must produce a hex placeholder rather than crash or
    silently drop the action. Ygopro-core may add new races over time."""
    from yugioh_core.constants import MSG_ANNOUNCE_RACE

    mapper = ActionMapper()
    # bit 50 — well outside any current RACE_NAMES entry
    mapper.update({"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": 1 << 50})
    assert mapper.num_actions == 1
    assert mapper.actions[0]["meta"]["label"] == f"Race(0x{1 << 50:x})"


def test_chain_pass_action_has_no_meta():
    """The pass action (category=1) must not carry meta — the describer falls
    through to the legacy 'Pass (no chain)' label when meta is absent."""
    from yugioh_core.constants import MSG_SELECT_CHAIN

    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 0,
            "chains": [
                {
                    "code": 12345,
                    "controller": 0,
                    "location": 0x10,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0xABCDEF,
                    "client_mode": 0,
                },
            ],
        }
    )
    assert mapper.num_actions == 2  # 1 chain + 1 pass
    assert mapper.actions[0]["meta"]["kind"] == "chain_link"
    assert mapper.actions[1].get("meta") is None
    assert mapper.actions[1]["category"] == 1


def test_counter_skips_cards_with_zero_counters():
    """Cards with counter_count=0 must NOT produce actions — they have nothing to remove.
    Existing extractor already filters these; this test guards against future regression."""
    from yugioh_core.constants import MSG_SELECT_COUNTER

    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_COUNTER,
            "player": 0,
            "counter_type": 0x1,
            "count": 2,
            "cards": [
                {"code": 111, "controller": 0, "location": 0x4, "sequence": 0, "counter_count": 3},
                {"code": 222, "controller": 0, "location": 0x4, "sequence": 1, "counter_count": 0},
                {"code": 333, "controller": 0, "location": 0x4, "sequence": 2, "counter_count": 5},
            ],
        }
    )
    assert mapper.num_actions == 2
    assert {a["code"] for a in mapper.actions} == {111, 333}


@pytest.mark.parametrize(
    "setup",
    [
        # (msg_dict, kind_string, has_card_code_flag)
        (
            {"msg_type": MSG_ANNOUNCE_NUMBER, "player": 0, "numbers": [3]},
            "number",
            False,
        ),
        (
            {"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": 0x1},
            "race",
            False,
        ),
        (
            {"msg_type": MSG_ANNOUNCE_ATTRIB, "player": 0, "available": 0x1},
            "attribute",
            False,
        ),
        (
            {"msg_type": MSG_ROCK_PAPER_SCISSORS, "player": 0},
            "rps",
            False,
        ),
        (
            {"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [0xDEAD]},
            "option",
            False,
        ),
        (
            {
                "msg_type": MSG_SELECT_CHAIN,
                "player": 0,
                "forced": 1,
                "chains": [
                    {
                        "code": 555,
                        "controller": 0,
                        "location": 0x10,
                        "sequence": 0,
                        "position": 0,
                        "desc": 0x1,
                        "client_mode": 0,
                    }
                ],
            },
            "chain_link",
            True,
        ),
        (
            {
                "msg_type": MSG_SELECT_COUNTER,
                "player": 0,
                "counter_type": 0x1,
                "count": 1,
                "cards": [
                    {
                        "code": 666,
                        "controller": 0,
                        "location": 0x4,
                        "sequence": 0,
                        "counter_count": 1,
                    }
                ],
            },
            "counter",
            True,
        ),
        # Card-bearing effect (IDLE_ACTIVATE)
        (
            {
                "msg_type": MSG_SELECT_IDLECMD,
                "player": 0,
                "summonable": [],
                "sp_summonable": [],
                "repositionable": [],
                "mset": [],
                "sset": [],
                "activatable": [
                    {
                        "code": 444,
                        "controller": 0,
                        "location": 0x4,
                        "sequence": 0,
                        "desc": 0x1,
                        "client_mode": 0,
                    }
                ],
                "to_bp": 0,
                "to_ep": 0,
                "shuffle_hand": 0,
            },
            "effect",
            True,
        ),
    ],
)
def test_action_meta_card_code_consistency(setup):
    """§6 invariant: for each kind, action_feats[2:6] code matches meta.extras['card_code']
    iff the kind is card-bearing. This catches drift between the feature vector and the
    meta dict — the two sources of truth for downstream consumers."""
    msg, expected_kind, has_card_code = setup
    mapper = ActionMapper()
    mapper.update(msg)
    feats = mapper.get_action_features()
    assert mapper.num_actions >= 1, f"Expected at least one action for {expected_kind}"

    for i, action in enumerate(mapper.actions):
        meta = action.get("meta")
        if meta is None:
            continue  # pass actions etc.
        assert meta["kind"] == expected_kind
        feats_row = list(feats[i])
        decoded_code = decode_u32(feats_row, 2)
        if has_card_code:
            assert decoded_code != 0, (
                f"{expected_kind} action #{i}: feats code is 0 but kind references a card"
            )
            assert "card_code" in meta.get("extras", {}), (
                f"{expected_kind} action #{i}: meta.extras missing card_code"
            )
            assert decoded_code == meta["extras"]["card_code"], (
                f"{expected_kind} action #{i}: feats code {decoded_code} != "
                f"meta.extras.card_code {meta['extras']['card_code']}"
            )
        else:
            assert decoded_code == 0, (
                f"{expected_kind} action #{i}: feats code is non-zero "
                f"but this kind has no associated card"
            )
            assert "card_code" not in meta.get("extras", {}), (
                f"{expected_kind} action #{i}: meta.extras must not have card_code "
                f"for card-less kinds"
            )


def test_announce_number_response_is_index_not_value():
    """The engine reads MSG_ANNOUNCE_NUMBER's response as an INDEX into the
    options list, not the announced value (third_party/ygopro-core/playerop.cpp:1109).
    Sending the value (e.g. 3 for [3,2,1]) makes the engine see 3 >= len(options)=3
    and emit MSG_RETRY → silent forfeit. This regression test pins the index semantics."""
    import struct

    from yugioh_core.constants import MSG_ANNOUNCE_NUMBER

    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_ANNOUNCE_NUMBER, "player": 0, "numbers": [3, 2, 1]})
    # Action 0 is "Announce 3" — the engine must receive index=0, NOT value=3.
    assert mapper.action_to_response(0) == struct.pack("<i", 0)
    assert mapper.action_to_response(1) == struct.pack("<i", 1)
    assert mapper.action_to_response(2) == struct.pack("<i", 2)


def test_announce_attrib_multi_step_two_picks_produces_or_mask():
    """Multi-bit AnnounceAttribute (count=2): step-by-step picks accumulate
    via _selected; the second pick's response packs the OR'd mask."""
    import struct

    from yugioh_core.constants import (
        ATTRIBUTE_DARK,
        ATTRIBUTE_LIGHT,
        ATTRIBUTE_WIND,
    )

    mapper = ActionMapper()
    msg = {
        "msg_type": MSG_ANNOUNCE_ATTRIB,
        "player": 0,
        "count": 2,
        "available": ATTRIBUTE_DARK | ATTRIBUTE_LIGHT | ATTRIBUTE_WIND,
    }

    # Step 1: 3 actions (one per available bit), all intermediate (no build_response)
    mapper.update(msg)
    assert mapper.num_actions == 3
    for i in range(3):
        assert mapper.action_to_response(i) is None

    # Pick action 0 — its index is the bit position of WIND.
    wind_bit = mapper.get_action_index(0)
    assert (1 << wind_bit) == ATTRIBUTE_WIND  # action.index encodes the bit position

    # Step 2: env injects _selected=[<wind_bit>]
    mapper.update({**msg, "_selected": [wind_bit]})
    # Now 2 actions remain (LIGHT and DARK), both terminal.
    assert mapper.num_actions == 2
    dark_bit = ATTRIBUTE_DARK.bit_length() - 1  # ATTRIBUTE_DARK is a single bit
    dark_action_pos = next(i for i, a in enumerate(mapper.actions) if a["index"] == dark_bit)
    response = mapper.action_to_response(dark_action_pos)
    # Engine reads uint32 mask: ATTRIBUTE_WIND | ATTRIBUTE_DARK
    assert response == struct.pack("<I", ATTRIBUTE_WIND | ATTRIBUTE_DARK)


def test_announce_attrib_count_one_emits_terminal_picks():
    """count=1 (the common case): each available bit becomes a terminal action,
    not an intermediate pick."""
    import struct

    from yugioh_core.constants import ATTRIBUTE_DARK

    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_ANNOUNCE_ATTRIB,
            "player": 0,
            "count": 1,
            "available": ATTRIBUTE_DARK,
        }
    )
    assert mapper.num_actions == 1
    response = mapper.action_to_response(0)
    assert response == struct.pack("<I", ATTRIBUTE_DARK)


# --- Test C: action controller relativization (engine-absolute → 0=agent/1=opp) ---


@pytest.mark.parametrize("agent_player", [0, 1])
def test_action_controller_relativizes_per_agent_player(agent_player):
    """SELECT_CHAIN with chain entries on both engine sides: the encoded
    controller byte must be 0 when the chain belongs to agent_player and
    1 otherwise, regardless of which engine player the agent is.

    Pins the contract that extractors read msg["_agent_player"] and
    relativize the engine-absolute controller into agent-relative form.
    """
    mapper = ActionMapper()
    mapper.update(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": agent_player,
            "_agent_player": agent_player,
            "forced": 1,  # no "no chain" option, just the entries
            "chains": [
                # Chain on engine player 0
                {
                    "flag": 0,
                    "code": 100,
                    "controller": 0,
                    "location": 0x04,
                    "sequence": 0,
                    "subsequence": 0,
                    "position": 0x01,
                    "desc": 0,
                },
                # Chain on engine player 1
                {
                    "flag": 0,
                    "code": 200,
                    "controller": 1,
                    "location": 0x04,
                    "sequence": 0,
                    "subsequence": 0,
                    "position": 0x01,
                    "desc": 0,
                },
            ],
        }
    )
    features = mapper.get_action_features()
    # New 28-byte layout: byte 6 = controller (relativized: 0=agent, 1=opp)
    ctrl_p0 = int(features[0][6])  # chain on engine player 0
    ctrl_p1 = int(features[1][6])  # chain on engine player 1
    if agent_player == 0:
        assert ctrl_p0 == 0  # agent's own
        assert ctrl_p1 == 1  # opponent's
    else:
        assert ctrl_p0 == 1  # opponent's (engine 0)
        assert ctrl_p1 == 0  # agent's own (engine 1)


# --- Test 1 (consolidated 8-case): single-byte wire field roundtrip ---
# Each case sets ONE wire field on a synthesized prompt, decodes the
# corresponding feats byte (per the 28-byte layout in _encode_action),
# and asserts the value survived. Catches drop/typo bugs in extractors.


def _idle_msg_with_card(**card_overrides) -> dict:
    """SELECT_IDLECMD with a single summonable card carrying the given fields."""
    card = {"code": 100, "controller": 0, "location": 0x02, "sequence": 0}
    card.update(card_overrides)
    return {
        "msg_type": MSG_SELECT_IDLECMD,
        "player": 0,
        "_agent_player": 0,
        "summonable": [card],
        "sp_summonable": [],
        "repositionable": [],
        "mset": [],
        "sset": [],
        "activatable": [],
        "to_bp": False,
        "to_ep": False,
        "shuffle_hand": False,
    }


def _battle_msg_with_attackable(**card_overrides) -> dict:
    """SELECT_BATTLECMD with a single attackable card."""
    card = {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "direct_attackable": 0}
    card.update(card_overrides)
    return {
        "msg_type": MSG_SELECT_BATTLECMD,
        "player": 0,
        "_agent_player": 0,
        "activatable": [],
        "attackable": [card],
        "to_m2": False,
        "to_ep": False,
    }


def _tribute_msg(**card_overrides) -> dict:
    """SELECT_TRIBUTE with a single card; release_param overridable."""
    card = {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "release_param": 1}
    card.update(card_overrides)
    return {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "_agent_player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [card],
    }


def _sum_msg(**card_overrides) -> dict:
    """SELECT_SUM with a single optional card; param overridable."""
    card = {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "param": 1}
    card.update(card_overrides)
    return {
        "msg_type": MSG_SELECT_SUM,
        "player": 0,
        "_agent_player": 0,
        "select_type": 0,
        "target_sum": card["param"] & 0xFFFF,
        "min": 1,
        "max": 1,
        "must_cards": [],
        "optional_cards": [card],
    }


def _card_msg(**card_overrides) -> dict:
    """SELECT_CARD with a single card; subsequence overridable."""
    card = {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "subsequence": 0}
    card.update(card_overrides)
    return {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "_agent_player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [card],
    }


def _chain_msg_with_chain(**chain_overrides) -> dict:
    """SELECT_CHAIN with one forced chain entry; position overridable."""
    chain = {
        "flag": 0,
        "code": 100,
        "controller": 0,
        "location": 0x04,
        "sequence": 0,
        "subsequence": 0,
        "position": 0x01,
        "desc": 0,
    }
    chain.update(chain_overrides)
    return {
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "_agent_player": 0,
        "forced": 1,
        "chains": [chain],
    }


def _counter_msg(**msg_overrides) -> dict:
    """SELECT_COUNTER with one card carrying counters."""
    card = {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "counter_count": 3}
    base = {
        "msg_type": MSG_SELECT_COUNTER,
        "player": 0,
        "_agent_player": 0,
        "counter_type": 0x01,
        "count": 1,
        "cards": [card],
    }
    base.update(msg_overrides)
    return base


@pytest.mark.parametrize(
    "case_name, msg_factory, byte_idx, expected",
    [
        # controller: opponent's card (engine ctrl=1, agent_player=0) → byte[6] == 1
        ("controller", lambda: _idle_msg_with_card(controller=1), 6, 1),
        # direct_attackable: byte[12] == 1
        ("direct_attackable", lambda: _battle_msg_with_attackable(direct_attackable=1), 12, 1),
        # release_param (tribute weight): byte[13] == 7
        ("release_param", lambda: _tribute_msg(release_param=7), 13, 7),
        # sum.param (sum weight): byte[13] == 4
        ("sum_param", lambda: _sum_msg(param=4), 13, 4),
        # subsequence: byte[10] == 2
        ("subsequence", lambda: _card_msg(subsequence=2), 10, 2),
        # position: byte[11] == 0x04 (FU-Def)
        ("position", lambda: _chain_msg_with_chain(position=0x04), 11, 0x04),
        # counter_type: byte[14] == 0x05 (low byte of counter_type=5)
        ("counter_type", lambda: _counter_msg(counter_type=5), 14, 5),
        # counter_count: byte[15] == 1 (n_remove = min(card.counter_count=3, msg.count=1))
        ("counter_count", lambda: _counter_msg(), 15, 1),
    ],
)
def test_extractor_single_byte_field_roundtrip(case_name, msg_factory, byte_idx, expected):
    """Each wire field shows up in the right byte of the encoded action."""
    mapper = ActionMapper()
    mapper.update(msg_factory())
    features = mapper.get_action_features()
    assert features[0][byte_idx] == expected, (
        f"case {case_name}: feats[0][{byte_idx}]={int(features[0][byte_idx])}, expected {expected}"
    )


# --- Test 2: desc decomposition (bytes 20-27 as u64 LE) ---


def test_extractor_desc_packs_into_bytes_20_27():
    """desc is encoded as u64 LE in bytes 20-27. Pack a known value with
    distinguishable passcode/n halves and verify decomposition."""
    desc_value = (0x12345 << 20) | 0x67  # passcode=0x12345, n=0x67
    msg = {
        "msg_type": MSG_SELECT_YESNO,
        "player": 0,
        "_agent_player": 0,
        "desc": desc_value,
    }
    mapper = ActionMapper()
    mapper.update(msg)
    feat = mapper.get_action_features()[0]
    decoded = 0
    for i in range(8):
        decoded |= int(feat[20 + i]) << (8 * i)
    assert decoded == desc_value
    assert (decoded >> 20) == 0x12345  # passcode half
    assert (decoded & 0xFFFFF) == 0x67  # n half (low 20 bits)


# --- Test 3: sequence widening regression (u8 → u16) ---


def test_extractor_sequence_widens_to_u16():
    """Sequence values >255 must survive encoding (deck/banished/GY targeting).
    Pre-fix would truncate sequence=300 to 44 (300 & 0xFF)."""
    msg = _card_msg(sequence=300)
    mapper = ActionMapper()
    mapper.update(msg)
    feat = mapper.get_action_features()[0]
    assert decode_u16(feat, 8) == 300


def test_extract_sort_actions_first_step_no_selected():
    """With no prior picks, emit N actions, all with build_response=None."""
    msg = {
        "msg_type": MSG_SORT_CARD,
        "player": 0,
        "cards": [
            {"code": 1, "controller": 0, "location": 0x01, "sequence": 0},
            {"code": 2, "controller": 0, "location": 0x01, "sequence": 1},
            {"code": 3, "controller": 0, "location": 0x01, "sequence": 2},
        ],
        "_agent_player": 0,
    }
    actions = _ACTION_EXTRACTORS[MSG_SORT_CARD](msg)

    assert len(actions) == 3
    assert all(a["build_response"] is None for a in actions)
    assert [a["index"] for a in actions] == [0, 1, 2]
    assert [a["code"] for a in actions] == [1, 2, 3]
    assert all(a.get("num_selected") == 1 for a in actions)


def test_extract_sort_actions_intermediate_step():
    """After one pick, emit N-1 actions for remaining cards, all None."""
    msg = {
        "msg_type": MSG_SORT_CARD,
        "player": 0,
        "cards": [
            {"code": 1, "controller": 0, "location": 0x01, "sequence": 0},
            {"code": 2, "controller": 0, "location": 0x01, "sequence": 1},
            {"code": 3, "controller": 0, "location": 0x01, "sequence": 2},
        ],
        "_selected": [1],
        "_agent_player": 0,
    }
    actions = _ACTION_EXTRACTORS[MSG_SORT_CARD](msg)

    assert len(actions) == 2
    assert [a["index"] for a in actions] == [0, 2]
    assert all(a["build_response"] is None for a in actions)
    assert all(a.get("num_selected") == 2 for a in actions)


def test_extract_sort_actions_final_step_builds_response():
    """On the final pick, build_response must produce the full permutation."""
    msg = {
        "msg_type": MSG_SORT_CARD,
        "player": 0,
        "cards": [
            {"code": 1, "controller": 0, "location": 0x01, "sequence": 0},
            {"code": 2, "controller": 0, "location": 0x01, "sequence": 1},
            {"code": 3, "controller": 0, "location": 0x01, "sequence": 2},
        ],
        "_selected": [1, 0],
        "_agent_player": 0,
    }
    actions = _ACTION_EXTRACTORS[MSG_SORT_CARD](msg)

    assert len(actions) == 1
    a = actions[0]
    assert a["index"] == 2
    assert a["build_response"] is not None
    assert a["build_response"]() == bytes([1, 0, 2])


def test_extract_sort_chain_uses_same_extractor():
    """MSG_SORT_CHAIN uses the same extractor as MSG_SORT_CARD."""
    assert _ACTION_EXTRACTORS[MSG_SORT_CHAIN] is _ACTION_EXTRACTORS[MSG_SORT_CARD]


def test_sort_card_multi_step_dispatch():
    """Drive _extract_sort_actions through a 3-pick sequence and verify
    the final bytes match the expected permutation."""
    base_msg = {
        "msg_type": MSG_SORT_CARD,
        "player": 0,
        "cards": [
            {"code": 10, "controller": 0, "location": 0x01, "sequence": 0},
            {"code": 20, "controller": 0, "location": 0x01, "sequence": 1},
            {"code": 30, "controller": 0, "location": 0x01, "sequence": 2},
        ],
        "_agent_player": 0,
    }

    selected: list[int] = []
    for pick in [2, 0, 1]:
        msg = {**base_msg, "_selected": list(selected)}
        actions = _ACTION_EXTRACTORS[MSG_SORT_CARD](msg)
        action = next(a for a in actions if a["index"] == pick)
        selected.append(action["index"])
        if action["build_response"] is not None:
            wire = action["build_response"]()
            assert selected == [2, 0, 1]
            assert wire == bytes([2, 0, 1])
            return
    raise AssertionError("build_response was never set; final pick did not finalize")
