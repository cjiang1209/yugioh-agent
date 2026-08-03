"""Unit tests for ActionDescriber.describe_prompt().

Uses ActionMapper.update() with synthetic parsed messages — no live duel needed.
"""

import pytest

from tests.env.conftest import obs_from_msg as _obs_from_msg
from yugioh_core.constants import (
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_DISFIELD,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    MSG_SORT_CARD,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
)
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.server.yugioh_environment import _build_prompt_meta


class FakeCardDB:
    """Minimal CardDatabase stand-in for tests."""

    def __init__(
        self,
        names: dict[int, str] | None = None,
        strings: dict[tuple[int, int], str] | None = None,
    ):
        self._names = names or {}
        self._strings = strings or {}

    def get_card_name(self, code: int) -> str:
        return self._names.get(code, f"Card#{code}")

    def get_card_string(self, passcode: int, n: int) -> str | None:
        return self._strings.get((passcode, n))


@pytest.fixture
def card_db():
    return FakeCardDB({89631139: "Blue-Eyes White Dragon", 46986414: "Dark Magician"})


def test_effectyn(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 2,
            "sequence": 0,
            "position": 0,
            "desc": 0,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "effect_yn"
    assert prompt["card_code"] == 89631139
    assert prompt["card_name"] == "Blue-Eyes White Dragon"
    assert prompt["desc"] == 0


def test_effectyn_preserves_nonzero_desc(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 2,
            "sequence": 0,
            "position": 0,
            "desc": 0x55,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == 0x55


def test_yesno(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "yes_no"
    assert "card_code" not in prompt
    assert prompt["desc"] == 0


def test_yesno_preserves_nonzero_desc(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0x1234,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == 0x1234


def test_select_card(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 2,
            "cards": [
                {"code": 89631139, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0},
                {"code": 46986414, "controller": 0, "location": 2, "sequence": 1, "subsequence": 0},
            ],
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "select_card"
    assert prompt["min"] == 1
    assert prompt["max"] == 2
    assert prompt["cancelable"] is False
    assert prompt["selected_count"] == 0


def test_select_card_with_selected(card_db):
    obs = _obs_from_msg(
        {
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
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "select_card"
    assert prompt["selected_count"] == 1


def test_tribute(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_TRIBUTE,
            "player": 0,
            "cancelable": 1,
            "min": 2,
            "max": 2,
            "cards": [
                {
                    "code": 89631139,
                    "controller": 0,
                    "location": 4,
                    "sequence": 0,
                    "release_param": 2,
                },
                {
                    "code": 46986414,
                    "controller": 0,
                    "location": 4,
                    "sequence": 1,
                    "release_param": 1,
                },
            ],
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
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


def test_tribute_with_selected(card_db):
    """A single monster with release_param=2 satisfies min_release=2."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_TRIBUTE,
            "player": 0,
            "cancelable": 0,
            "min": 2,
            "max": 2,
            "_selected": [0],
            "cards": [
                {
                    "code": 89631139,
                    "controller": 0,
                    "location": 4,
                    "sequence": 0,
                    "release_param": 2,
                },
                {
                    "code": 46986414,
                    "controller": 0,
                    "location": 4,
                    "sequence": 1,
                    "release_param": 1,
                },
            ],
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "tribute"
    assert prompt["release_total"] == 2
    assert prompt["cards_selected"] == 1


def test_unselect_card(card_db):
    obs = _obs_from_msg(
        {
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
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "select_card"
    assert prompt["min"] == 1
    assert prompt["max"] == 3
    assert prompt["finishable"] is True
    assert "selected_count" not in prompt


def test_chain(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "spe_count": 0,
            "forced": 0,
            "hint_timing": 0,
            "other_timing": 0,
            "chains": [
                {
                    "code": 46986414,
                    "controller": 0,
                    "location": 8,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "chain_link"
    assert prompt["forced"] is False


def test_chain_forced(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "spe_count": 0,
            "forced": 1,
            "hint_timing": 0,
            "other_timing": 0,
            "chains": [
                {
                    "code": 46986414,
                    "controller": 0,
                    "location": 8,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["forced"] is True


def test_position(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_POSITION,
            "player": 0,
            "code": 89631139,
            "positions": POS_FACEUP_ATTACK | POS_FACEDOWN_DEFENSE,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "position"
    assert prompt["card_code"] == 89631139
    assert prompt["card_name"] == "Blue-Eyes White Dragon"


def test_idle_cmd(card_db):
    obs = _obs_from_msg(
        {
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
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "idle_cmd"
    assert len(prompt) == 1  # only "type", no extra fields


def test_unknown_msg_type(card_db):
    obs = _obs_from_msg({"msg_type": 9999})
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "unknown"


def test_place(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_PLACE,
            "player": 0,
            "count": 1,
            "field_mask": 0x1F,  # 5 monster zones
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "place"
    assert prompt["count"] == 1


def test_disfield_maps_to_place(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_DISFIELD,
            "player": 0,
            "count": 1,
            "field_mask": 0x1F,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["type"] == "place"


def test_prompt_type_map_includes_announce_kinds():
    """describe_prompt returns the correct `type` strings for the five prompts
    that previously fell through to 'unknown'."""
    from yugioh_core.constants import (
        MSG_ANNOUNCE_ATTRIB,
        MSG_ANNOUNCE_NUMBER,
        MSG_ANNOUNCE_RACE,
        MSG_ROCK_PAPER_SCISSORS,
        MSG_SELECT_COUNTER,
    )

    class _StubDB:
        def get_card_name(self, code):
            return f"Card{code}"

    cases = [
        ({"msg_type": MSG_ANNOUNCE_NUMBER, "player": 0, "numbers": [3]}, "number"),
        ({"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": 0x1}, "race"),
        ({"msg_type": MSG_ANNOUNCE_ATTRIB, "player": 0, "available": 0x1}, "attribute"),
        ({"msg_type": MSG_ROCK_PAPER_SCISSORS, "player": 0}, "rps"),
        (
            {
                "msg_type": MSG_SELECT_COUNTER,
                "player": 0,
                "counter_type": 0x1,
                "count": 1,
                "cards": [],
            },
            "counter",
        ),
        ({"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [0x1]}, "option"),
    ]
    for msg, expected_type in cases:
        obs = _obs_from_msg(msg)
        describer = ActionDescriber(_StubDB(), sys_strings=None)
        result = describer.describe_prompt(obs)
        assert result["type"] == expected_type, (
            f"msg_type={msg['msg_type']}: expected '{expected_type}', got '{result['type']}'"
        )


def test_yesno_resolves_sysstring_to_prompt_text(card_db):
    sys_strings = {0x42: "Activate effect?"}
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0x42,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=sys_strings)
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == 0x42
    assert prompt["prompt_text"] == "Activate effect?"


def test_yesno_zero_desc_yields_null_prompt_text(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0,
        }
    )
    describer = ActionDescriber(card_db, sys_strings={0x42: "Activate?"})
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == 0
    assert prompt["prompt_text"] is None


def test_yesno_unknown_desc_yields_null_prompt_text(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0x999,
        }
    )
    describer = ActionDescriber(card_db, sys_strings={0x42: "Activate?"})
    prompt = describer.describe_prompt(obs)
    assert prompt["prompt_text"] is None


def test_yesno_no_resolver_yields_null_prompt_text(card_db):
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0x42,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=None)
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == 0x42
    assert prompt["prompt_text"] is None


def test_effectyn_resolves_card_string_to_prompt_text():
    """EFFECTYN desc encoding: (passcode << 20) | n. Resolver calls
    card_db.get_card_string(passcode, n)."""
    card_db = FakeCardDB(
        names={89631139: "Blue-Eyes White Dragon"},
        strings={(89631139, 5): "Negate the attack?"},
    )
    desc = (89631139 << 20) | 5
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 2,
            "sequence": 0,
            "position": 0,
            "desc": desc,
        }
    )
    describer = ActionDescriber(card_db, sys_strings={})
    prompt = describer.describe_prompt(obs)
    assert prompt["desc"] == desc
    assert prompt["prompt_text"] == "Negate the attack?"


def test_effectyn_substitutes_card_name_and_location_in_template(card_db):
    """Two-`%ls` template: first placeholder → card name, second → location.
    `location: 2` is LOCATION_HAND."""
    sys_strings = {221: 'Activate the Trigger Effect of "%ls" from [%ls]?'}
    desc = 221
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 2,
            "sequence": 0,
            "position": 0,
            "desc": desc,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=sys_strings)
    prompt = describer.describe_prompt(obs)
    assert prompt["prompt_text"] == (
        'Activate the Trigger Effect of "Blue-Eyes White Dragon" from [Hand]?'
    )


def test_yesno_with_single_placeholder_template_drops_to_null(card_db):
    """YESNO has no card_code/location, so a `%ls` template can't be
    filled — prompt_text falls back to None so the client synthesizes."""
    sys_strings = {95: 'Use the effect of "%ls"?'}
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 95,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=sys_strings)
    prompt = describer.describe_prompt(obs)
    assert prompt["prompt_text"] is None


def test_prompt_text_drops_to_null_when_format_specifier_remains(card_db):
    """A `%d`-bearing template we don't know how to fill yields None."""
    sys_strings = {204: 'Remove %d "%ls"'}
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 2,
            "sequence": 0,
            "position": 0,
            "desc": 204,
        }
    )
    describer = ActionDescriber(card_db, sys_strings=sys_strings)
    prompt = describer.describe_prompt(obs)
    assert prompt["prompt_text"] is None


def test_build_prompt_meta_sort_card_first_step():
    class FakeMapper:
        msg_type = MSG_SORT_CARD
        msg = {
            "msg_type": MSG_SORT_CARD,
            "player": 0,
            "cards": [
                {"code": 100, "controller": 0, "location": 0x01, "sequence": 0},
                {"code": 200, "controller": 0, "location": 0x01, "sequence": 1},
                {"code": 300, "controller": 0, "location": 0x01, "sequence": 2},
            ],
        }

    meta = _build_prompt_meta(FakeMapper())
    assert meta["msg_type"] == MSG_SORT_CARD
    assert meta["count"] == 3
    assert meta["picked_cards"] == []


def test_build_prompt_meta_sort_card_intermediate_step():
    class FakeMapper:
        msg_type = MSG_SORT_CARD
        msg = {
            "msg_type": MSG_SORT_CARD,
            "player": 0,
            "cards": [
                {"code": 100, "controller": 0, "location": 0x01, "sequence": 0},
                {"code": 200, "controller": 0, "location": 0x01, "sequence": 1},
                {"code": 300, "controller": 0, "location": 0x01, "sequence": 2},
            ],
            "_selected": [1],
        }

    meta = _build_prompt_meta(FakeMapper())
    assert meta["count"] == 3
    assert meta["picked_cards"] == [{"code": 200, "location": 0x01}]


def test_build_prompt_meta_select_card_picked_cards():
    class FakeMapper:
        msg_type = MSG_SELECT_CARD
        msg = {
            "msg_type": MSG_SELECT_CARD,
            "min": 1,
            "max": 2,
            "cancelable": 0,
            "cards": [
                {"code": 100, "controller": 0, "location": 0x01, "sequence": 0},
                {"code": 200, "controller": 0, "location": 0x01, "sequence": 1},
                {"code": 300, "controller": 0, "location": 0x01, "sequence": 2},
            ],
            "_selected": [2, 0],
        }

    meta = _build_prompt_meta(FakeMapper())
    assert meta["selected_count"] == 2
    assert meta["picked_cards"] == [
        {"code": 300, "location": 0x01},
        {"code": 100, "location": 0x01},
    ]
    assert meta["selected"] == [2, 0]


def test_build_prompt_meta_select_tribute_picked_cards():
    class FakeMapper:
        msg_type = MSG_SELECT_TRIBUTE
        msg = {
            "msg_type": MSG_SELECT_TRIBUTE,
            "min": 1,
            "max": 2,
            "cancelable": 0,
            "cards": [
                {"code": 100, "controller": 0, "location": 0x04, "sequence": 0, "release_param": 1},
                {"code": 200, "controller": 0, "location": 0x04, "sequence": 1, "release_param": 1},
            ],
            "_selected": [1],
        }

    meta = _build_prompt_meta(FakeMapper())
    assert meta["cards_selected"] == 1
    assert meta["picked_cards"] == [{"code": 200, "location": 0x04}]
    assert meta["selected"] == [1]


def test_build_prompt_meta_select_unselect_picked_cards():
    """For MSG_SELECT_UNSELECT_CARD, picked cards come from `unselectable`."""

    class FakeMapper:
        msg_type = MSG_SELECT_UNSELECT_CARD
        msg = {
            "msg_type": MSG_SELECT_UNSELECT_CARD,
            "finishable": 1,
            "min": 1,
            "max": 3,
            "selectable": [
                {"code": 300, "controller": 0, "location": 0x01, "sequence": 2},
            ],
            "unselectable": [
                {"code": 100, "controller": 0, "location": 0x01, "sequence": 0},
                {"code": 200, "controller": 0, "location": 0x01, "sequence": 1},
            ],
        }

    meta = _build_prompt_meta(FakeMapper())
    assert meta["picked_cards"] == [
        {"code": 100, "location": 0x01},
        {"code": 200, "location": 0x01},
    ]


def test_effectyn_prompt_meta_has_relativized_controller() -> None:
    """Opponent's card, agent is player 1 -> controller must read 0 (mine)."""
    msg = {
        "msg_type": MSG_SELECT_EFFECTYN,
        "code": 999,
        "controller": 1,
        "location": 0x04,
        "sequence": 2,
        "desc": 0,
    }
    obs = _obs_from_msg(msg, agent_player=1)
    pm = obs.prompt_meta
    assert pm["controller"] == 0  # absolute 1 == agent 1 -> "mine"
    assert pm["sequence"] == 2

    obs0 = _obs_from_msg(msg, agent_player=0)
    assert obs0.prompt_meta["controller"] == 1  # same card, other seat


def test_sum_prompt_meta_branch_exists() -> None:
    # min=2 deliberately differs from _build_prompt_meta's own default (1),
    # so a mutation that drops the assignment and falls through to the
    # default cannot coincidentally pass.
    msg = {
        "msg_type": MSG_SELECT_SUM,
        "select_type": 0,
        "target_sum": 8,
        "min": 2,
        "max": 2,
        "must_cards": [],
        "optional_cards": [],
    }
    pm = _obs_from_msg(msg).prompt_meta
    assert pm["target_sum"] == 8
    assert pm["select_type"] == 0
    assert pm["min"] == 2 and pm["max"] == 2
    assert pm["must_cards"] == []


def test_sum_prompt_meta_selected_and_must_cards_fields() -> None:
    """`selected` (non-empty) and per-card `must_cards` fields (param != 0,
    controller relativized) for MSG_SELECT_SUM. Checked at both seats so an
    inverted `_relativize_controller` call cannot hide behind a player-0-only
    assertion."""

    def _meta(agent_player: int) -> dict:
        class FakeMapper:
            msg_type = MSG_SELECT_SUM
            msg = {
                "msg_type": MSG_SELECT_SUM,
                "select_type": 0,
                "target_sum": 8,
                "min": 1,
                "max": 2,
                "_agent_player": agent_player,
                "_selected": [0],
                "must_cards": [
                    {"code": 555, "controller": 1, "location": 0x04, "sequence": 3, "param": 7},
                ],
                "optional_cards": [],
            }

        return _build_prompt_meta(FakeMapper())

    meta0 = _meta(0)
    assert meta0["selected"] == [0]
    assert meta0["must_cards"] == [
        {"code": 555, "controller": 1, "location": 0x04, "sequence": 3, "param": 7}
    ]

    meta1 = _meta(1)
    # Same absolute controller (1), but now == agent_player -> relativizes to 0.
    assert meta1["must_cards"] == [
        {"code": 555, "controller": 0, "location": 0x04, "sequence": 3, "param": 7}
    ]
