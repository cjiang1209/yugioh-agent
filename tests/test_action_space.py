"""Test action space mapping."""

import struct

import numpy as np
import pytest

from yugioh_env.action_space import ActionMapper, MAX_ACTIONS, ACTION_FEATURES
from yugioh_env.constants import (
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_SELECT_CARD,
    MSG_SELECT_PLACE,
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
            {"flag": 0, "code": 100, "controller": 0, "location": 2,
             "sequence": 0, "subsequence": 0, "desc": 0},
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
