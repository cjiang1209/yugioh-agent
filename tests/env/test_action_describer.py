"""Tests for action_describer — tribute multi-step descriptions."""

from yugioh_core.constants import MSG_SELECT_TRIBUTE
from yugioh_env.action_space import ActionMapper
from yugioh_env.server.action_describer import describe_actions


class _StubCardDB:
    """Minimal stub that avoids any SQLite dependency."""

    def get_card_name(self, code: int) -> str:
        return f"Card{code}"


def test_tribute_describe_finish():
    """Tribute finish action gets proper description, card picks get 'tribute' category."""
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

    # Pick release_param=2 card to reach finish-eligible state
    resp = mapper.action_to_response(0)
    assert resp is None
    mapper.update({**msg, "_selected": [0]})

    # Now we have 1 card pick + 1 finish
    assert mapper.num_actions == 2

    descs = describe_actions(mapper, _StubCardDB())

    # Card pick action
    assert descs[0]["category"] == "tribute"
    assert "Tribute" in descs[0]["description"]

    # Finish action
    assert descs[1]["category"] == "finish"
    assert descs[1]["description"] == "Finish tributing (1 card)"


import pytest


def _make_mapper_with_meta(msg):
    from yugioh_env.action_space import ActionMapper
    m = ActionMapper()
    m.update(msg)
    return m


@pytest.mark.parametrize("msg, expected_desc, expected_category", [
    (
        {"msg_type": 143, "player": 0, "numbers": [3]},  # MSG_ANNOUNCE_NUMBER
        "Announce 3", "number",
    ),
    (
        {"msg_type": 140, "player": 0, "available": 0x1},  # MSG_ANNOUNCE_RACE (Warrior)
        "Warrior", "race",
    ),
    (
        {"msg_type": 141, "player": 0, "available": 0x20},  # MSG_ANNOUNCE_ATTRIB (DARK)
        "DARK", "attribute",
    ),
    (
        {"msg_type": 132, "player": 0},  # MSG_ROCK_PAPER_SCISSORS, first action
        "Rock", "rps",
    ),
    (
        {"msg_type": 14, "player": 0, "options": [0xabc]},  # MSG_SELECT_OPTION
        "effect 0xabc", "option",
    ),
])
def test_describer_uses_meta_label_for_simple_kinds(msg, expected_desc, expected_category):
    """For each kind whose msg_type branch reads meta directly, verify the
    label and category come through unchanged."""
    mapper = _make_mapper_with_meta(msg)
    descs = describe_actions(mapper, _StubCardDB())
    assert descs[0]["description"] == expected_desc
    assert descs[0]["category"] == expected_category


def test_describer_rewrites_counter_with_card_name():
    """Counter description combines meta.extras.counter_count (from extractor)
    with card_name (from DB, only available in describer)."""
    from yugioh_core.constants import MSG_SELECT_COUNTER
    mapper = _make_mapper_with_meta({
        "msg_type": MSG_SELECT_COUNTER, "player": 0,
        "counter_type": 0x1, "count": 2,
        "cards": [{"code": 999, "controller": 0, "location": 0x4,
                   "sequence": 0, "counter_count": 3}],
    })
    descs = describe_actions(mapper, _StubCardDB())
    assert descs[0]["description"] == "Remove 2 from Card999"
    assert descs[0]["category"] == "counter"


def test_describer_meta_field_passes_through_as_none_when_absent():
    """For prompts whose extractor doesn't emit meta (e.g. SELECT_YESNO), the
    result row's meta field is None — no fabricated meta from the describer."""
    from yugioh_core.constants import MSG_SELECT_YESNO
    mapper = _make_mapper_with_meta({"msg_type": MSG_SELECT_YESNO, "player": 0})
    descs = describe_actions(mapper, _StubCardDB())
    assert descs[0]["meta"] is None
    assert descs[0]["description"] == "Yes"
