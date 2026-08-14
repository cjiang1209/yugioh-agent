"""Tests for ActionDescriber — uses observation-shaped inputs."""

import pytest

from tests.env.conftest import MINIMAL_MSGS
from tests.env.conftest import obs_from_msg as _obs_from_msg
from yugioh_core.constants import (
    ATTRIBUTE_DARK,
    LOCATION_DECK,
    LOCATION_GRAVE,
    LOCATION_MZONE,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_RACE,
    MSG_ROCK_PAPER_SCISSORS,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_COUNTER,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    MSG_SORT_CARD,
    RACE_WARRIOR,
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
    assert obs.num_actions == 2

    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)

    # Card pick action
    assert details[0].category == "tribute"
    assert "Tribute" in details[0].description

    # Finish action
    assert details[1].category == "finish"
    assert details[1].description == "Finish tributing (1 card)"


@pytest.mark.parametrize(
    "msg, expected_desc, expected_category",
    [
        (
            {"msg_type": MSG_ANNOUNCE_NUMBER, "player": 0, "numbers": [3]},
            "Announce 3",
            "number",
        ),
        (
            {"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": RACE_WARRIOR},
            "Warrior",
            "race",
        ),
        (
            {"msg_type": MSG_ANNOUNCE_ATTRIB, "player": 0, "available": ATTRIBUTE_DARK},
            "DARK",
            "attribute",
        ),
        (
            {"msg_type": MSG_ROCK_PAPER_SCISSORS, "player": 0},  # first action
            "Rock",
            "rps",
        ),
        (
            {"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [0xABC]},
            "effect 0xabc",
            "option",
        ),
    ],
)
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

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_COUNTER,
            "player": 0,
            "counter_type": 0x1,
            "count": 2,
            "cards": [
                {
                    "code": 999,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "counter_count": 3,
                }
            ],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].description == "Remove 2 from Card999"
    assert details[0].category == "counter"


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

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_OPTION,
            "player": 0,
            "options": [0xABC, 0xDEF],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})  # gives an empty StringResolver
    describer._text._resolver = _StubResolver({0xABC: "Special Summon a Spellcaster"})
    details = describer.describe_all(obs)
    # Resolved option uses the real string.
    assert details[0].description == "Special Summon a Spellcaster"
    # Unresolved option falls back to the placeholder.
    assert details[1].description == "effect 0xdef"


def test_describer_chain_appends_resolved_effect_text():
    """When chain meta resolves AND a card_name is known, append the effect text;
    otherwise fall back to today's `Chain {card_name}` form."""
    from yugioh_core.constants import MSG_SELECT_CHAIN

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 1,
            "chains": [
                {
                    "code": 777,
                    "controller": 0,
                    "location": LOCATION_GRAVE,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0x123,
                    "client_mode": 0,
                }
            ],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._text._resolver = _StubResolver({0x123: "Increase ATK by 1000"})
    details = describer.describe_all(obs)
    assert details[0].description == "Chain Card777: Increase ATK by 1000"


def test_describer_chain_falls_back_when_resolver_returns_none():
    """Unresolved chain desc keeps today's `Chain {card_name}` form (no trailing colon)."""
    from yugioh_core.constants import MSG_SELECT_CHAIN

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 1,
            "chains": [
                {
                    "code": 777,
                    "controller": 0,
                    "location": LOCATION_GRAVE,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0x123,
                    "client_mode": 0,
                }
            ],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._text._resolver = _StubResolver({})
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

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 1,
            # code=0 disables card_name lookup in describer's `card_name = card_db.get_card_name(code) if code else ""`
            "chains": [
                {
                    "code": 0,
                    "controller": 0,
                    "location": LOCATION_GRAVE,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0x123,
                    "client_mode": 0,
                }
            ],
        }
    )
    describer = ActionDescriber(_NoNameDB(), sys_strings={})
    describer._text._resolver = _StubResolver({0x123: "Real effect"})
    details = describer.describe_all(obs)
    # Resolved text dropped; falls back to anonymous "Chain #0".
    assert details[0].description == "Chain #0"


def test_describer_idle_activate_appends_resolved_effect_text():
    """When an idle ACTIVATE action's meta resolves, append `: {effect}` to
    the existing `Activate {card_name}` label."""
    from yugioh_core.constants import MSG_SELECT_IDLECMD

    obs = _obs_from_msg(
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
                    "code": 555,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "desc": 0xABC,
                    "client_mode": 0,
                }
            ],
            "to_bp": 0,
            "to_ep": 0,
            "shuffle_hand": 0,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._text._resolver = _StubResolver({0xABC: "Increase ATK"})
    details = describer.describe_all(obs)
    assert details[0].description == "Activate Card555: Increase ATK"
    assert details[0].category == "activate"


def test_idle_summon_zero_code_card_has_no_trailing_space():
    """A zero-code summonable entry (card_name resolves to "") must not leave
    a trailing space in the label. The old `_dispatch` guarded with
    `if code and card_name:`; the new CardCommand case must preserve that."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_IDLECMD,
            "player": 0,
            "summonable": [{"code": 0, "controller": 0, "location": LOCATION_MZONE, "sequence": 0}],
            "sp_summonable": [],
            "repositionable": [],
            "mset": [],
            "sset": [],
            "activatable": [],
            "to_bp": 0,
            "to_ep": 0,
            "shuffle_hand": 0,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].description == "Normal Summon"


def test_describer_effectyn_yes_is_plain_yes():
    """EFFECTYN action labels are plain Yes/No; card name and resolved
    effect text live on the prompt header, not on the action label."""
    from yugioh_core.constants import MSG_SELECT_EFFECTYN

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 777,
            "controller": 0,
            "location": LOCATION_MZONE,
            "sequence": 0,
            "desc": 0xDEF,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._text._resolver = _StubResolver({0xDEF: "Special Summon"})
    details = describer.describe_all(obs)
    assert details[0].description == "Yes"
    assert details[1].description == "No"


def test_describer_yesno_yes_is_plain_yes():
    """YESNO has no card context; both actions are plain Yes/No.
    The resolved question lives on prompt_text."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0x111,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings={})
    describer._text._resolver = _StubResolver({0x111: "Pay LP"})
    details = describer.describe_all(obs)
    assert details[0].description == "Yes"
    assert details[1].description == "No"


def test_describe_all_returns_one_per_legal_action():
    """describe_all returns one ActionDetails per legal action."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0,
        }
    )
    # SELECT_YESNO produces 2 legal actions (Yes, No).
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert len(details) == 2
    assert obs.num_actions == 2  # sanity: matches the action count


def test_describe_action_sort_card():
    """Sort-card actions should describe as 'Place {card} next' with category 'sort'."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SORT_CARD,
            "player": 0,
            "cards": [
                {"code": 100, "controller": 0, "location": LOCATION_DECK, "sequence": 0},
                {"code": 200, "controller": 0, "location": LOCATION_DECK, "sequence": 1},
                {"code": 300, "controller": 0, "location": LOCATION_DECK, "sequence": 2},
            ],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert len(details) == 3
    for d in details:
        assert d.category == "sort"
        assert d.description.startswith("Place ")


def test_describe_raises_past_the_last_legal_action():
    """With one descriptor per legal action there are no inactive slots to ask
    about, only indices past the end."""
    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_YESNO,
            "player": 0,
            "desc": 0,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    # YESNO offers exactly two actions, so 2 is already past the end.
    assert obs.num_actions == 2
    with pytest.raises(IndexError, match="out of range"):
        describer.describe(obs, 2)
    with pytest.raises(IndexError, match="out of range"):
        describer.describe(obs, MAX_ACTIONS + 10)
    with pytest.raises(IndexError, match="out of range"):
        describer.describe(obs, -1)


def test_describer_announce_card_uses_card_name():
    """MSG_ANNOUNCE_CARD action resolves the declared card's name from the DB."""
    from yugioh_core.constants import MSG_ANNOUNCE_CARD, OPCODE_ISCODE

    obs = _obs_from_msg(
        {
            "msg_type": MSG_ANNOUNCE_CARD,
            "player": 0,
            "opcodes": [777, OPCODE_ISCODE],
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert details[0].description == "Declare Card777"
    assert details[0].category == "announce_card"


def test_effectyn_details_equal_at_both_seats():
    """Re-sourcing from prompt_meta must reproduce today's values exactly."""
    msg = {
        "msg_type": MSG_SELECT_EFFECTYN,
        "player": 0,
        "code": 999,
        "controller": 1,
        "location": LOCATION_MZONE,
        "sequence": 2,
        "desc": 0,
    }
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    for seat, expected_ctrl in ((0, 1), (1, 0)):
        d = describer.describe(_obs_from_msg(msg, agent_player=seat), 0)
        assert d.card_code == 999
        assert d.controller == expected_ctrl
        assert d.location == LOCATION_MZONE
        assert d.sequence == 2


def test_shared_variant_labels_differ_by_msg_type():
    """PickCard under TRIBUTE vs CARD must not collapse to one wording.

    This is the regression variant-only dispatch would cause.
    """
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    tribute = describer.describe(
        _obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_TRIBUTE], "msg_type": MSG_SELECT_TRIBUTE}), 0
    ).description
    select = describer.describe(
        _obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_CARD], "msg_type": MSG_SELECT_CARD}), 0
    ).description
    assert tribute.startswith("Tribute")
    assert select.startswith("Select")

    # The Pass variant is shared too: MSG_SELECT_CHAIN's "no chain" pass and
    # MSG_SELECT_UNSELECT_CARD's "finish selection" pass must not collapse to
    # the same wording (or swap categories) just because both are `Pass()`.
    chain_pass = describer.describe(
        _obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_CHAIN], "msg_type": MSG_SELECT_CHAIN}), 1
    )
    assert chain_pass.description == "Pass (no chain)"
    assert chain_pass.category == "pass"

    unselect_finish = describer.describe(
        _obs_from_msg(
            {**MINIMAL_MSGS[MSG_SELECT_UNSELECT_CARD], "msg_type": MSG_SELECT_UNSELECT_CARD}
        ),
        1,
    )
    assert unselect_finish.description == "Finish selection"
    assert unselect_finish.category == "finish"


def test_place_details_report_the_real_seat():
    """Value change #1: controller 0 -> real relativized seat."""
    obs = _obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_PLACE], "msg_type": MSG_SELECT_PLACE})
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    seats = {describer.describe(obs, i).controller for i in range(obs.num_actions)}
    assert seats == {0, 1}


def test_sort_details_report_parsed_coordinates():
    """Value change #2: location/sequence fabricated 0 -> parsed.

    A second card at a non-zero sequence is included so the assertion can't
    coincide with the harness's fabricated-zero default (MINIMAL_MSGS' lone
    `_CARD` has sequence=0, which would make this vacuous on its own).
    """
    msg = {
        **MINIMAL_MSGS[MSG_SORT_CARD],
        "msg_type": MSG_SORT_CARD,
        "cards": [
            *MINIMAL_MSGS[MSG_SORT_CARD]["cards"],
            {"code": 4321, "controller": 0, "location": LOCATION_MZONE, "sequence": 3},
        ],
    }
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(_obs_from_msg(msg))
    d = details[1]
    assert d.location == LOCATION_MZONE
    assert d.sequence == 3


def test_counter_details_report_parsed_coordinates():
    """Value change #3 — the one omitted from earlier drafts.

    Also pins the CardRef -> ActionDetails wiring for `controller` and
    `sequence` (the describer's main re-sourcing path): the card is given
    controller=1 (relativized to 1 at agent_player=0) and a non-zero
    sequence so neither field can coincide with a fabricated 0.
    """
    msg = {
        **MINIMAL_MSGS[MSG_SELECT_COUNTER],
        "msg_type": MSG_SELECT_COUNTER,
        "cards": [
            {
                "code": 1234,
                "controller": 1,
                "location": LOCATION_MZONE,
                "sequence": 5,
                "counter_count": 3,
            }
        ],
    }
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    d = describer.describe(_obs_from_msg(msg, agent_player=0), 0)
    assert d.controller == 1
    assert d.location == LOCATION_MZONE
    assert d.sequence == 5


def test_action_details_to_dict_has_no_meta_key():
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    d = describer.describe(
        _obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_IDLECMD], "msg_type": MSG_SELECT_IDLECMD}), 0
    )
    assert set(d.to_dict()) == {
        "index",
        "description",
        "category",
        "card_code",
        "card_name",
        "controller",
        "location",
        "sequence",
    }


def test_announce_race_falls_back_to_hex_for_unknown_race():
    """An unmapped race bit must produce a hex placeholder rather than crash or
    silently drop the action. Ygopro-core may add new races over time."""
    from yugioh_core.constants import MSG_ANNOUNCE_RACE

    # bit 50 — well outside any current RACE_NAMES entry
    obs = _obs_from_msg({"msg_type": MSG_ANNOUNCE_RACE, "player": 0, "available": 1 << 50})
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    assert len(details) == 1
    assert details[0].description == f"Race(0x{1 << 50:x})"


def test_idle_phase_change_labels():
    """No existing test asserts the actual rendered phase-change labels; a
    to_bp<->to_ep transposition in _IDLE_DESCS or the PhaseChange case's
    `cat` lookup would ship a wrong label ("To End Phase" on the to_bp slot)
    with the rest of the suite green.

    Anchors each description to the descriptor's own `to` field (ground
    truth from action_space.py, pinned separately in
    test_idle_phase_change_to_values) rather than grouping by the
    describer's OWN output category -- a bug that moves category and label
    together (e.g. swapping which category constant a `to` value maps to)
    would otherwise still produce internally-consistent (category,
    description) pairs and slip through a category-keyed assertion.
    """
    from yugioh_env.models import PhaseChange

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
            "to_bp": 1,
            "to_ep": 1,
            "shuffle_hand": 0,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    bp_idx = next(
        i
        for i, d in enumerate(obs.action_descriptors)
        if isinstance(d, PhaseChange) and d.to == "bp"
    )
    ep_idx = next(
        i
        for i, d in enumerate(obs.action_descriptors)
        if isinstance(d, PhaseChange) and d.to == "ep"
    )
    assert details[bp_idx].description == "To Battle Phase"
    assert details[ep_idx].description == "To End Phase"


def test_battle_phase_change_labels():
    """Battle-side sibling of test_idle_phase_change_labels: a to_m2<->to_ep
    transposition would ship "To End Phase" on the to_m2 slot undetected."""
    from yugioh_core.constants import MSG_SELECT_BATTLECMD
    from yugioh_env.models import PhaseChange

    obs = _obs_from_msg(
        {
            "msg_type": MSG_SELECT_BATTLECMD,
            "player": 0,
            "activatable": [],
            "attackable": [],
            "to_m2": 1,
            "to_ep": 1,
        }
    )
    describer = ActionDescriber(_StubCardDB(), sys_strings=None)
    details = describer.describe_all(obs)
    m2_idx = next(
        i
        for i, d in enumerate(obs.action_descriptors)
        if isinstance(d, PhaseChange) and d.to == "m2"
    )
    ep_idx = next(
        i
        for i, d in enumerate(obs.action_descriptors)
        if isinstance(d, PhaseChange) and d.to == "ep"
    )
    assert details[m2_idx].description == "To Main Phase 2"
    assert details[ep_idx].description == "To End Phase"
