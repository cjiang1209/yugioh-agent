"""Test action space mapping."""

import numpy as np
import pytest

from yugioh_env.action_space import ActionMapper, MAX_ACTIONS, ACTION_FEATURES
from yugioh_env.constants import (
    MSG_SELECT_YESNO,
    MSG_SELECT_CHAIN,
    MSG_SELECT_POSITION,
    MSG_SELECT_OPTION,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)


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


def test_invalid_action_index():
    """Invalid action index should raise ValueError."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    with pytest.raises(ValueError):
        mapper.action_to_response(5)
