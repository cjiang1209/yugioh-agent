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
