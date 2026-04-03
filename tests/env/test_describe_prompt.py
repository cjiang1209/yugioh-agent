"""Unit tests for describe_prompt().

Uses ActionMapper.update() with synthetic parsed messages — no live duel needed.
"""

import pytest

from yugioh_core.constants import (
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,
    MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_POSITION,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_PLACE,
    MSG_SELECT_DISFIELD,
    MSG_SELECT_OPTION,
    POS_FACEUP_ATTACK,
    POS_FACEDOWN_DEFENSE,
)
from yugioh_env.action_space import ActionMapper
from yugioh_env.server.action_describer import describe_prompt


class FakeCardDB:
    """Minimal CardDatabase stand-in for tests."""

    def __init__(self, names: dict[int, str] | None = None):
        self._names = names or {}

    def get_card_name(self, code: int) -> str:
        return self._names.get(code, f"Card#{code}")


@pytest.fixture
def mapper():
    return ActionMapper()


@pytest.fixture
def card_db():
    return FakeCardDB({89631139: "Blue-Eyes White Dragon", 46986414: "Dark Magician"})


def test_effectyn(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_EFFECTYN,
        "player": 0,
        "code": 89631139,
        "controller": 0,
        "location": 2,
        "sequence": 0,
        "position": 0,
        "desc": 0,
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "effect_yn"
    assert prompt["card_code"] == 89631139
    assert prompt["card_name"] == "Blue-Eyes White Dragon"


def test_yesno(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_YESNO,
        "player": 0,
        "desc": 0,
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "yes_no"
    assert "card_code" not in prompt


def test_select_card(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 2,
        "cards": [
            {"code": 89631139, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 46986414, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "select_card"
    assert prompt["min"] == 1
    assert prompt["max"] == 2
    assert prompt["cancelable"] is False
    assert prompt["selected_count"] == 0


def test_select_card_with_selected(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "_selected": [0],
        "cards": [
            {"code": 89631139, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
            {"code": 46986414, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "select_card"
    assert prompt["selected_count"] == 1


def test_tribute(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 1,
        "min": 2,
        "max": 2,
        "cards": [
            {"code": 89631139, "controller": 0, "location": 4, "sequence": 0, "release_param": 2},
            {"code": 46986414, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "tribute"
    assert prompt["min_release"] == 2
    assert prompt["max_cards"] == 2
    assert prompt["cancelable"] is True
    assert prompt["release_total"] == 0
    assert prompt["cards_selected"] == 0
    # min/max/selected_count should NOT be present (tribute has its own fields)
    assert "min" not in prompt
    assert "max" not in prompt
    assert "selected_count" not in prompt


def test_tribute_with_selected(mapper, card_db):
    """A single monster with release_param=2 satisfies min_release=2."""
    mapper.update({
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 2,
        "max": 2,
        "_selected": [0],
        "cards": [
            {"code": 89631139, "controller": 0, "location": 4, "sequence": 0, "release_param": 2},
            {"code": 46986414, "controller": 0, "location": 4, "sequence": 1, "release_param": 1},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "tribute"
    assert prompt["release_total"] == 2
    assert prompt["cards_selected"] == 1


def test_unselect_card(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_UNSELECT_CARD,
        "player": 0,
        "finishable": 1,
        "cancelable": 0,
        "min": 1,
        "max": 3,
        "selectable": [
            {"code": 89631139, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
        ],
        "unselectable": [],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "select_card"
    assert prompt["min"] == 1
    assert prompt["max"] == 3
    assert prompt["finishable"] is True
    assert "selected_count" not in prompt


def test_chain(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "spe_count": 0,
        "forced": 0,
        "hint_timing": 0,
        "other_timing": 0,
        "chains": [
            {"code": 46986414, "controller": 0, "location": 8, "sequence": 0,
             "position": 0, "desc": 0, "client_mode": 0},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "chain"
    assert prompt["forced"] is False


def test_chain_forced(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "spe_count": 0,
        "forced": 1,
        "hint_timing": 0,
        "other_timing": 0,
        "chains": [
            {"code": 46986414, "controller": 0, "location": 8, "sequence": 0,
             "position": 0, "desc": 0, "client_mode": 0},
        ],
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["forced"] is True


def test_position(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_POSITION,
        "player": 0,
        "code": 89631139,
        "positions": POS_FACEUP_ATTACK | POS_FACEDOWN_DEFENSE,
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "position"
    assert prompt["card_code"] == 89631139
    assert prompt["card_name"] == "Blue-Eyes White Dragon"


def test_idle_cmd(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_IDLECMD,
        "player": 0,
        "summonable": [],
        "sp_summonable": [],
        "repositionable": [],
        "mset": [],
        "sset": [],
        "activatable": [],
        "to_bp": 0,
        "to_ep": 1,
        "shuffle_hand": 0,
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "idle_cmd"
    assert len(prompt) == 1  # only "type", no extra fields


def test_unknown_msg_type(mapper, card_db):
    mapper.update({"msg_type": 9999})
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "unknown"


def test_place(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_PLACE,
        "player": 0,
        "count": 1,
        "field_mask": 0x1F,  # 5 monster zones
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "place"
    assert prompt["count"] == 1


def test_disfield_maps_to_place(mapper, card_db):
    mapper.update({
        "msg_type": MSG_SELECT_DISFIELD,
        "player": 0,
        "count": 1,
        "field_mask": 0x1F,
    })
    prompt = describe_prompt(mapper, card_db)
    assert prompt["type"] == "place"
