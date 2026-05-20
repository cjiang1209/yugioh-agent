"""Tests for ActionDescriber — uses observation-shaped inputs."""

import pytest

from tests.env.conftest import obs_from_msg as _obs_from_msg
from yugioh_core.constants import (
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_YESNO,
)
from yugioh_core.encoding import MAX_ACTIONS
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.action_space import ActionMapper


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

    obs = _obs_from_msg(msg, _selected=[0])

    # Now we have 1 card pick + 1 finish
    assert sum(obs.action_mask) == 2

    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)

    # Card pick action
    assert details[0].category == "tribute"
    assert "Tribute" in details[0].description

    # Finish action
    assert details[1].category == "finish"
    assert details[1].description == "Finish tributing (1 card)"


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
    obs = _obs_from_msg(msg)
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].description == expected_desc
    assert details[0].category == expected_category


def test_describer_rewrites_counter_with_card_name():
    """Counter description combines meta.extras.counter_count (from extractor)
    with card_name (from DB, only available in describer)."""
    from yugioh_core.constants import MSG_SELECT_COUNTER
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_COUNTER, "player": 0,
        "counter_type": 0x1, "count": 2,
        "cards": [{"code": 999, "controller": 0, "location": 0x4,
                   "sequence": 0, "counter_count": 3}],
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].description == "Remove 2 from Card999"
    assert details[0].category == "counter"


def test_describer_meta_field_passes_through_as_none_when_absent():
    """For prompts whose extractor doesn't emit meta (e.g. SELECT_PLACE), the
    result row's meta field is None — no fabricated meta from the describer."""
    from yugioh_core.constants import MSG_SELECT_PLACE
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_PLACE, "player": 0,
        "count": 1, "field_mask": 0,  # all 32 zones unblocked → many actions
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].meta is None


class _StubResolver:
    """Resolves a fixed dict of desc_u64 → string. Anything else returns None."""

    def __init__(self, table: dict[int, str]):
        self._table = table

    def resolve(self, desc_u64: int) -> str | None:
        return self._table.get(desc_u64)


def test_describer_option_uses_resolver_when_provided():
    """When a resolver returns a real string for the option's raw_value, the
    describer prefers it over the placeholder `effect 0x...` label."""
    from yugioh_core.constants import MSG_SELECT_OPTION
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_OPTION, "player": 0, "options": [0xabc, 0xdef],
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})  # gives an empty StringResolver
    describer._resolver = _StubResolver({0xabc: "Special Summon a Spellcaster"})
    details = describer.describe_all(obs)
    # Resolved option uses the real string.
    assert details[0].description == "Special Summon a Spellcaster"
    # Unresolved option falls back to the placeholder.
    assert details[1].description == "effect 0xdef"


def test_describer_chain_appends_resolved_effect_text():
    """When chain meta resolves AND a card_name is known, append the effect text;
    otherwise fall back to today's `Chain {card_name}` form."""
    from yugioh_core.constants import MSG_SELECT_CHAIN
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_CHAIN, "player": 0, "forced": 1,
        "chains": [{"code": 777, "controller": 0, "location": 0x10,
                    "sequence": 0, "position": 0, "desc": 0x123, "client_mode": 0}],
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._resolver = _StubResolver({0x123: "Increase ATK by 1000"})
    details = describer.describe_all(obs)
    assert details[0].description == "Chain Card777: Increase ATK by 1000"


def test_describer_chain_falls_back_when_resolver_returns_none():
    """Unresolved chain desc keeps today's `Chain {card_name}` form (no trailing colon)."""
    from yugioh_core.constants import MSG_SELECT_CHAIN
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_CHAIN, "player": 0, "forced": 1,
        "chains": [{"code": 777, "controller": 0, "location": 0x10,
                    "sequence": 0, "position": 0, "desc": 0x123, "client_mode": 0}],
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._resolver = _StubResolver({})
    details = describer.describe_all(obs)
    assert details[0].description == "Chain Card777"


def test_describer_chain_drops_resolved_text_when_card_name_missing():
    """When card_name is empty (e.g. anonymous chain entry), the describer
    intentionally drops the resolved effect text rather than emit awkward
    `Chain : <effect>`. Chain falls back to the index form `Chain #N`."""
    from yugioh_core.constants import MSG_SELECT_CHAIN

    class _NoNameDB:
        def get_card_name(self, code: int) -> str:
            return ""  # simulate missing/anonymous card

    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_CHAIN, "player": 0, "forced": 1,
        # code=0 disables card_name lookup in describer's `card_name = card_db.get_card_name(code) if code else ""`
        "chains": [{"code": 0, "controller": 0, "location": 0x10,
                    "sequence": 0, "position": 0, "desc": 0x123, "client_mode": 0}],
    })
    describer = ActionDescriber(_NoNameDB(), sys_strings={})
    describer._resolver = _StubResolver({0x123: "Real effect"})
    details = describer.describe_all(obs)
    # Resolved text dropped; falls back to anonymous "Chain #0".
    assert details[0].description == "Chain #0"


def test_describer_idle_activate_appends_resolved_effect_text():
    """When an idle ACTIVATE action's meta resolves, append `: {effect}` to
    the existing `Activate {card_name}` label."""
    from yugioh_core.constants import MSG_SELECT_IDLECMD
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_IDLECMD, "player": 0,
        "summonable": [], "sp_summonable": [], "repositionable": [],
        "mset": [], "sset": [],
        "activatable": [{"code": 555, "controller": 0, "location": 0x4,
                         "sequence": 0, "desc": 0xabc, "client_mode": 0}],
        "to_bp": 0, "to_ep": 0, "shuffle_hand": 0,
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._resolver = _StubResolver({0xabc: "Increase ATK"})
    details = describer.describe_all(obs)
    assert details[0].description == "Activate Card555: Increase ATK"
    assert details[0].category == "activate"


def test_describer_effectyn_yes_is_plain_yes():
    """EFFECTYN action labels are plain Yes/No; card name and resolved
    effect text live on the prompt header, not on the action label."""
    from yugioh_core.constants import MSG_SELECT_EFFECTYN
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_EFFECTYN, "player": 0,
        "code": 777, "controller": 0, "location": 0x4, "sequence": 0,
        "desc": 0xdef,
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._resolver = _StubResolver({0xdef: "Special Summon"})
    details = describer.describe_all(obs)
    assert details[0].description == "Yes"
    assert details[1].description == "No"
    assert details[0].meta is None
    assert details[1].meta is None


def test_describer_yesno_yes_is_plain_yes():
    """YESNO has no card context; both actions are plain Yes/No.
    The resolved question lives on prompt_text."""
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0x111,
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._resolver = _StubResolver({0x111: "Pay LP"})
    details = describer.describe_all(obs)
    assert details[0].description == "Yes"
    assert details[1].description == "No"
    assert details[0].meta is None
    assert details[1].meta is None


def test_describe_all_returns_one_per_legal_action():
    """describe_all returns N descriptors when N action_mask bits are set."""
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0,
    })
    # SELECT_YESNO produces 2 legal actions (Yes, No).
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert len(details) == 2
    assert sum(obs.action_mask) == 2  # sanity: matches the mask


def test_describe_raises_on_inactive_slot():
    """describe(idx) raises IndexError when the slot is inactive."""
    obs = _obs_from_msg({
        "msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0,
    })
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    # Slot 5 is inactive (only slots 0 and 1 are legal for YESNO).
    with pytest.raises(IndexError, match="inactive"):
        describer.describe(obs, 5)
    # Out-of-range slot also raises.
    with pytest.raises(IndexError, match="out of range"):
        describer.describe(obs, MAX_ACTIONS + 10)
