"""Test action space mapping."""

import struct

import numpy as np
import pytest

from yugioh_env.action_space import ActionMapper, MAX_ACTIONS, ACTION_FEATURES
from yugioh_core.constants import (
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_RACE,
    MSG_ROCK_PAPER_SCISSORS,
    MSG_SELECT_CARD,
    MSG_SELECT_COUNTER,
    MSG_SELECT_PLACE,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    MSG_SELECT_CHAIN,
    MSG_SELECT_POSITION,
    MSG_SELECT_OPTION,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_env import response_builder as rb


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
    mapper.update({
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "spe_count": 0,
        "forced": 0,
        "hint_timing": 0,
        "other_timing": 0,
        "chains": [
            {"code": 100, "controller": 0, "location": 2,
             "sequence": 0, "position": 0, "desc": 0, "client_mode": 0},
        ],
    })
    # 1 chain + 1 "no chain" = 2 actions
    assert mapper.num_actions == 2


def test_position_actions():
    """Position selection should list available positions."""
    mapper = ActionMapper()
    mapper.update({
        "msg_type": MSG_SELECT_POSITION,
        "player": 0,
        "code": 12345,
        "positions": POS_FACEUP_ATTACK | POS_FACEUP_DEFENSE,
    })
    assert mapper.num_actions == 2


def test_option_actions():
    """Option selection should produce correct number of actions."""
    mapper = ActionMapper()
    mapper.update({
        "msg_type": MSG_SELECT_OPTION,
        "player": 0,
        "options": [100, 200, 300],
    })
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
    mapper.update({
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [
            {"code": 89631139, "controller": 0, "location": 2, "sequence": 3, "subsequence": 0},
        ],
    })
    features = mapper.get_action_features()
    feat = features[0]
    # feat[0] = msg_type
    assert feat[0] == MSG_SELECT_CARD
    # feat[2:6] = code as uint32 LE (89631139 = 0x0557B1A3)
    code = int(feat[2]) | (int(feat[3]) << 8) | (int(feat[4]) << 16) | (int(feat[5]) << 24)
    assert code == 89631139
    # feat[6] = location, feat[7] = sequence, feat[8] = index
    assert feat[6] == 2   # location
    assert feat[7] == 3   # sequence
    assert feat[8] == 0   # index


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


def test_select_card_actions_response():
    """Full round-trip: MSG_SELECT_CARD -> ActionMapper -> response bytes."""
    mapper = ActionMapper()
    mapper.update({
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [
            {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 200, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
        ],
    })
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
    mapper.update({
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
    })
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
        assert features[i][9] == 1
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
    assert features2[3][9] == 1  # num_selected = 1 (accumulated so far)
    # Card actions have num_selected = 2 (will be 2 after this pick)
    for i in range(3):
        assert features2[i][9] == 2
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
    mapper.update({
        "msg_type": MSG_SELECT_COUNTER,
        "player": 0,
        "counter_type": 0x01,
        "count": 3,
        "cards": [
            {"code": 100, "controller": 0, "location": 4, "sequence": 0, "counter_count": 2},
            {"code": 200, "controller": 0, "location": 4, "sequence": 1, "counter_count": 5},
        ],
    })
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
        assert features[i][9] == 1

    # Step 2: after picking card 0
    mapper.update({**msg, "_selected": [0]})
    features2 = mapper.get_action_features()
    # Remaining card actions have num_selected = 2 (will be 2 after pick)
    for i in range(2):
        assert features2[i][9] == 2


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
    mapper.update({
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "forced": 0,
        "chains": [
            {"code": 12345, "controller": 0, "location": 0x10, "sequence": 0,
             "position": 0, "desc": 0xabcdef, "client_mode": 0},
        ],
    })
    assert mapper.num_actions == 2  # 1 chain + 1 pass
    assert mapper.actions[0]["meta"]["kind"] == "chain_link"
    assert mapper.actions[1].get("meta") is None
    assert mapper.actions[1]["category"] == 1


def test_counter_skips_cards_with_zero_counters():
    """Cards with counter_count=0 must NOT produce actions — they have nothing to remove.
    Existing extractor already filters these; this test guards against future regression."""
    from yugioh_core.constants import MSG_SELECT_COUNTER
    mapper = ActionMapper()
    mapper.update({
        "msg_type": MSG_SELECT_COUNTER,
        "player": 0,
        "counter_type": 0x1,
        "count": 2,
        "cards": [
            {"code": 111, "controller": 0, "location": 0x4, "sequence": 0, "counter_count": 3},
            {"code": 222, "controller": 0, "location": 0x4, "sequence": 1, "counter_count": 0},
            {"code": 333, "controller": 0, "location": 0x4, "sequence": 2, "counter_count": 5},
        ],
    })
    assert mapper.num_actions == 2
    assert {a["code"] for a in mapper.actions} == {111, 333}


@pytest.mark.parametrize("setup", [
    # (msg_dict, kind_string, has_card_code_flag)
    (
        {"msg_type": MSG_ANNOUNCE_NUMBER, "player": 0, "numbers": [3]},
        "number", False,
    ),
    (
        {"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": 0x1},
        "race", False,
    ),
    (
        {"msg_type": MSG_ANNOUNCE_ATTRIB, "player": 0, "available": 0x1},
        "attribute", False,
    ),
    (
        {"msg_type": MSG_ROCK_PAPER_SCISSORS, "player": 0},
        "rps", False,
    ),
    (
        {"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [0xdead]},
        "option", False,
    ),
    (
        {"msg_type": MSG_SELECT_CHAIN, "player": 0, "forced": 1,
         "chains": [{"code": 555, "controller": 0, "location": 0x10,
                     "sequence": 0, "position": 0, "desc": 0x1, "client_mode": 0}]},
        "chain_link", True,
    ),
    (
        {"msg_type": MSG_SELECT_COUNTER, "player": 0, "counter_type": 0x1, "count": 1,
         "cards": [{"code": 666, "controller": 0, "location": 0x4,
                    "sequence": 0, "counter_count": 1}]},
        "counter", True,
    ),
])
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
        # Decode card code from bytes [2:6] as uint32 (little-endian)
        decoded_code = (int(feats_row[2])
                        | (int(feats_row[3]) << 8)
                        | (int(feats_row[4]) << 16)
                        | (int(feats_row[5]) << 24))
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
        ATTRIBUTE_DARK, ATTRIBUTE_LIGHT, ATTRIBUTE_WIND,
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
    mapper.update({
        "msg_type": MSG_ANNOUNCE_ATTRIB,
        "player": 0,
        "count": 1,
        "available": ATTRIBUTE_DARK,
    })
    assert mapper.num_actions == 1
    response = mapper.action_to_response(0)
    assert response == struct.pack("<I", ATTRIBUTE_DARK)
