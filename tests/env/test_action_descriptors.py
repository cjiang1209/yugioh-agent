"""Unit tests for the ActionDescriptor tagged union and its production."""

from dataclasses import fields

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.env.conftest import MINIMAL_MSGS
from yugioh_core.constants import LOCATION_MZONE, MSG_SELECT_CARD, MSG_SELECT_IDLECMD
from yugioh_env.action_space import _ACTION_EXTRACTORS
from yugioh_env.models import ActionDescriptor, CardRef, Pass, PickCard
from yugioh_env.server.yugioh_environment import _build_action_descriptors

ADAPTER = TypeAdapter(ActionDescriptor)


def test_discriminator_round_trip_preserves_variant() -> None:
    src = PickCard(
        engine_index=2,
        num_selected=1,
        param=None,
        card=CardRef(code=7, controller=0, location=LOCATION_MZONE, sequence=1),
    )
    back = ADAPTER.validate_python(ADAPTER.dump_python(src))
    assert isinstance(back, PickCard)
    assert back.card.location == LOCATION_MZONE


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"kind": "nope"})


def test_pass_is_distinct_from_finish_pick() -> None:
    """Chain pass / unselect finish are fieldless; only the harness finish
    carries num_selected. Collapsing them would lose that distinction."""
    assert not hasattr(Pass(), "num_selected")
    assert {f.name for f in fields(Pass)} == {"kind"}


@pytest.mark.parametrize("msg_type", sorted(_ACTION_EXTRACTORS))
def test_producer_covers_every_extractor(msg_type: int) -> None:
    msg = {**MINIMAL_MSGS[msg_type], "msg_type": msg_type, "_agent_player": 0}
    actions = _ACTION_EXTRACTORS[msg_type](msg)
    out = _build_action_descriptors(actions)
    for i in range(len(actions)):
        assert out[i] is not None


def test_producer_raises_on_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown action kind"):
        _build_action_descriptors([{"kind": "bogus"}])


def test_producer_raises_on_missing_kind() -> None:
    with pytest.raises(ValueError, match="untagged action"):
        _build_action_descriptors([{"code": 1}])


@pytest.mark.parametrize("msg_type", sorted(_ACTION_EXTRACTORS))
def test_no_cardref_built_from_filler(msg_type: int) -> None:
    """location == 0 is not a valid bitmask; no variant may emit it."""
    msg = {**MINIMAL_MSGS[msg_type], "msg_type": msg_type, "_agent_player": 0}
    for d in _build_action_descriptors(_ACTION_EXTRACTORS[msg_type](msg)):
        if d is not None and getattr(d, "card", None) is not None:
            assert d.card.location != 0, f"{msg_type}: fabricated CardRef"


# Per-kind field checks comparing a built descriptor back against the raw
# action dict it was built from. This is deliberately generic (not tied to
# the literal values in MINIMAL_MSGS) so it catches any field mis-wiring in
# _DESCRIPTOR_BUILDERS -- e.g. narrowing counter_type to u8, inverting
# Confirm.yes, swapping ChoosePosition's position/card_code, or swapping
# PickBit's bit/mask (engine_index is the bit, value is 1<<bit).
_KIND_FIELD_CHECKS = {
    "pick_card": lambda d, a: (
        d.engine_index == a["index"]
        and d.num_selected == a.get("num_selected", 1)
        and d.param == a.get("param")
        and d.card.code == a.get("code", 0)
        and d.card.controller == a.get("controller", 0)
        and d.card.location == a["location"]
        and d.card.sequence == a["sequence"]
    ),
    "pick_bit": lambda d, a: d.engine_index == a["index"] and d.value == a["value"],
    "finish_pick": lambda d, a: d.num_selected == a["num_selected"],
    "card_command": lambda d, a: (
        d.engine_index == a["index"]
        and d.command == a["category"]
        and d.card.code == a.get("code", 0)
        and d.card.location == a["location"]
        and d.card.sequence == a["sequence"]
    ),
    "activate_effect": lambda d, a: (
        d.engine_index == a["index"]
        and d.desc == a.get("desc", 0)
        and d.card.location == a["location"]
    ),
    "attack": lambda d, a: (
        d.engine_index == a["index"]
        and d.direct_attackable == bool(a.get("direct_attackable", 0))
        and d.card.location == a["location"]
    ),
    "phase_change": lambda d, a: d.to == a["to"],
    "confirm": lambda d, a: d.yes == a["yes"] and d.desc == a.get("desc", 0),
    "choose_option": lambda d, a: d.engine_index == a["index"] and d.desc == a.get("desc", 0),
    "choose_position": lambda d, a: d.position == a["index"] and d.card_code == a.get("code", 0),
    "place_zone": lambda d, a: (
        d.controller == a["controller"]
        and d.location == a["location"]
        and d.sequence == a["sequence"]
    ),
    "announce_number": lambda d, a: d.engine_index == a["index"] and d.value == a["value"],
    "announce_card": lambda d, a: d.card_code == a.get("code", 0),
    "choose_rps": lambda d, a: d.choice == a["index"],
    "select_counter": lambda d, a: (
        d.engine_index == a["index"]
        and d.counter_type == a["counter_type"]
        and d.counter_count == a["counter_count"]
        and d.card.location == a["location"]
    ),
    "pass": lambda d, a: True,
}


@pytest.mark.parametrize("msg_type", sorted(_ACTION_EXTRACTORS))
def test_descriptor_fields_match_source_action(msg_type: int) -> None:
    msg = {**MINIMAL_MSGS[msg_type], "msg_type": msg_type, "_agent_player": 0}
    actions = _ACTION_EXTRACTORS[msg_type](msg)
    descriptors = _build_action_descriptors(actions)
    assert actions, f"MINIMAL_MSGS[{msg_type}] produced no actions"
    for action, descriptor in zip(actions, descriptors, strict=False):
        kind = action["kind"]
        check = _KIND_FIELD_CHECKS.get(kind)
        assert check is not None, f"no field check registered for kind {kind!r}"
        assert check(descriptor, action), (
            f"{kind} field mismatch for msg_type={msg_type}: "
            f"action={action!r} descriptor={descriptor!r}"
        )


def test_pick_card_num_selected_tracks_selection_progress() -> None:
    """PickCard.num_selected must be len(selected) + 1, not a hardcoded 1.

    MINIMAL_MSGS[MSG_SELECT_CARD] alone can't exercise this: with a single
    card and max=1, the first (only) pick always has num_selected == 1,
    which coincides with the builder's own `a.get("num_selected", 1)`
    fallback -- a hardcoded `num_selected=1` would pass unnoticed. Seed one
    card as already selected via `_selected` so the second card's pick
    action carries num_selected == 2.
    """
    msg = {
        "msg_type": MSG_SELECT_CARD,
        "_agent_player": 0,
        "player": 0,
        "cancelable": 0,
        "min": 0,
        "max": 2,
        "cards": [
            {"code": 1, "controller": 0, "location": LOCATION_MZONE, "sequence": 0},
            {"code": 2, "controller": 0, "location": LOCATION_MZONE, "sequence": 1},
        ],
        "_selected": [0],
    }
    actions = _ACTION_EXTRACTORS[MSG_SELECT_CARD](msg)
    picks = [a for a in actions if a["kind"] == "pick_card"]
    assert len(picks) == 1, f"expected exactly one remaining pick, got {picks!r}"
    action = picks[0]
    assert action["num_selected"] == 2, "fixture must exercise num_selected != 1"
    descriptor = _build_action_descriptors([action])[0]
    assert descriptor.num_selected == action["num_selected"]


def test_descriptor_none_iff_mask_zero() -> None:
    from tests.env.conftest import obs_from_msg

    obs = obs_from_msg(
        {
            "msg_type": MSG_SELECT_IDLECMD,
            "player": 0,
            "summonable": [
                {"code": 1234, "controller": 0, "location": LOCATION_MZONE, "sequence": 0}
            ],
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
    assert any(m for m in obs.action_mask)  # sanity: at least one legal action
    for i, m in enumerate(obs.action_mask):
        assert (obs.action_descriptors[i] is None) == (m == 0)
