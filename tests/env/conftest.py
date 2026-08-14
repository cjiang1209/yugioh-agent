"""Env-specific test fixtures (lib, script_dirs, duel)."""

from __future__ import annotations

from pathlib import Path

import pytest


def obs_from_action_count(num_legal: int = 3):
    """Build a YuGiOhObservation carrying `num_legal` legal actions.

    Legality is the descriptor list's length, so the legal slots are always
    the prefix 0..num_legal-1. `Pass` is the filler -- it encodes to a byte
    row without needing a prompt. For tests that drive an opponent's choice
    without needing a real board.
    """
    from yugioh_env.models import Pass, YuGiOhObservation

    return YuGiOhObservation(action_descriptors=[Pass()] * num_legal)


def action_features(mapper):
    """The packed action rows for the mapper's current prompt.

    Produced from a structured observation the way the server produces them,
    so callers exercise the one encoder there is.
    """
    from yugioh_env.models import YuGiOhObservation
    from yugioh_env.server.yugioh_environment import _build_action_descriptors, _build_prompt_meta
    from yugioh_rl.obs_encoder import encode_observation

    obs = YuGiOhObservation(
        action_descriptors=_build_action_descriptors(mapper.actions),
        prompt_meta=_build_prompt_meta(mapper),
    )
    return encode_observation(obs)["actions"]


def obs_from_msg(msg: dict, *, _selected: list[int] | None = None, agent_player: int = 0):
    """Build a YuGiOhObservation from a single SELECT message.

    Mirrors what the server's _make_observation produces for that message,
    using _build_action_descriptors and _build_prompt_meta to populate the
    parallel meta fields. The optional _selected list seeds the mapper's
    multi-step selection state for tests that need it (e.g., tribute finish
    actions). agent_player relativizes seat-dependent fields (e.g.
    controller) the same way the live server does.
    """
    from yugioh_env.action_space import ActionMapper
    from yugioh_env.models import YuGiOhObservation
    from yugioh_env.server.yugioh_environment import (
        _build_action_descriptors,
        _build_prompt_meta,
    )

    msg = {**msg, "_agent_player": agent_player}
    mapper = ActionMapper()
    mapper.update(msg)
    if _selected is not None:
        mapper.update({**msg, "_selected": _selected})
    return YuGiOhObservation(
        action_descriptors=_build_action_descriptors(mapper.actions),
        prompt_meta=_build_prompt_meta(mapper),
        events=[],
        done=False,
        reward=0.0,
    )


@pytest.fixture
def script_dirs(project_root) -> list[Path]:
    dirs = [
        project_root / "third_party" / "CardScripts" / "official",
        project_root / "third_party" / "CardScripts" / "pre-release",
        project_root / "third_party" / "CardScripts",
    ]
    existing = [d for d in dirs if d.exists()]
    if not existing:
        pytest.skip("CardScripts not found. Set up git submodules.")
    return existing


@pytest.fixture
def lib():
    """Load the OCG core library."""
    try:
        from yugioh_env.lib_loader import load_library

        return load_library()
    except FileNotFoundError:
        pytest.skip("libocgcore not found. Run: make build")


@pytest.fixture
def duel(lib, card_db, script_dirs, deck_path):
    """Create a Duel instance ready to use."""
    from yugioh_env.duel import Duel

    d = Duel(lib, card_db, script_dirs)
    yield d
    d.destroy()


# ─── MINIMAL_MSGS: one branch-complete msg per registered extractor ─────────
#
# Each entry is the smallest legal msg that exercises every `kind` branch its
# extractor can produce, per the field shapes in message_parser.py's `_parse_*`
# functions. Every list is load-bearing: the kind-coverage tests compare the
# exact set of kinds an extractor emits, so trimming one fails them. The
# comments below explain only non-obvious value choices. `desc` and option
# values are deliberately non-zero so a hardcoded 0 in the descriptor builders
# is caught rather than coinciding with a correct result.

from yugioh_core.constants import (
    COUNTER_NEED_ENABLE,
    LOCATION_MZONE,
    LOCATION_OVERLAY,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_CARD,
    MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_RACE,
    MSG_ROCK_PAPER_SCISSORS,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_COUNTER,
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
    MSG_SORT_CHAIN,
    OPCODE_ISCODE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)

_CARD = {"code": 1234, "controller": 0, "location": LOCATION_MZONE, "sequence": 0}

MINIMAL_MSGS: dict[int, dict] = {
    MSG_SELECT_IDLECMD: {
        "player": 0,
        "summonable": [_CARD],
        "sp_summonable": [_CARD],
        "repositionable": [_CARD],
        "mset": [_CARD],
        "sset": [_CARD],
        "activatable": [{**_CARD, "desc": 0x99, "client_mode": 0}],
        "to_bp": 1,
        "to_ep": 1,
        "shuffle_hand": 0,
    },
    MSG_SELECT_BATTLECMD: {
        "player": 0,
        "activatable": [{**_CARD, "desc": 0x99, "client_mode": 0}],
        "attackable": [{**_CARD, "direct_attackable": 0}],
        "to_m2": 1,
        "to_ep": 1,
    },
    MSG_SELECT_EFFECTYN: {
        "player": 0,
        "code": 1234,
        "controller": 0,
        "location": LOCATION_MZONE,
        "sequence": 0,
        "desc": 0x99,
    },
    MSG_SELECT_YESNO: {"player": 0, "desc": 0x99},
    # The option value itself becomes the action's `desc`.
    MSG_SELECT_OPTION: {"player": 0, "options": [7]},
    # min=0 < max=1 makes can_finish true immediately (selected=[]), so one
    # call yields both a pick and the harness finish action.
    MSG_SELECT_CARD: {
        "player": 0,
        "cancelable": 0,
        "min": 0,
        "max": 1,
        "cards": [_CARD],
    },
    # forced=0, so the pass action is offered alongside the chain link.
    MSG_SELECT_CHAIN: {
        "player": 0,
        "forced": 0,
        "chains": [{**_CARD, "desc": 0x99, "position": 0}],
    },
    # Inverted mask: field_mask=0 leaves every zone OPEN, so both my zones and
    # the opponent's are offered.
    MSG_SELECT_PLACE: {"player": 0, "count": 1, "field_mask": 0},
    MSG_SELECT_DISFIELD: {"player": 0, "count": 1, "field_mask": 0},
    MSG_SELECT_POSITION: {"player": 0, "code": 1234, "positions": 0x0F},
    # release_param=1 satisfies min on the first pick, but can_finish is
    # evaluated against the *pre-pick* selected=[] (total 0 < min 1), so no
    # finish action appears.
    MSG_SELECT_TRIBUTE: {
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [{**_CARD, "release_param": 1}],
    },
    # param == target_sum satisfies completes() on the first pick, but
    # can_finish requires min_sel <= 0, which min=1 rules out, so no finish
    # action appears.
    MSG_SELECT_SUM: {
        "player": 0,
        "select_type": 0,
        "target_sum": 4,
        "min": 1,
        "max": 1,
        "must_cards": [],
        "optional_cards": [{**_CARD, "param": 4}],
    },
    # finishable=1 offers the custom-coded finish alongside the picks.
    MSG_SELECT_UNSELECT_CARD: {
        "player": 0,
        "finishable": 1,
        "cancelable": 0,
        "min": 0,
        "max": 1,
        "selectable": [{**_CARD, "subsequence": 0}],
        "unselectable": [],
    },
    # _extract_sort_actions never sets can_finish, so no finish action appears.
    MSG_SORT_CARD: {"player": 0, "cards": [_CARD]},
    MSG_SORT_CHAIN: {"player": 0, "cards": [_CARD]},
    # can_finish is always False, so no finish action appears.
    MSG_ANNOUNCE_RACE: {"player": 0, "count": 1, "available": 1},
    MSG_ANNOUNCE_ATTRIB: {"player": 0, "count": 1, "available": 1},
    MSG_ANNOUNCE_NUMBER: {"player": 0, "numbers": [4]},
    # A literal ISCODE filter is what yields a declarable code.
    MSG_ANNOUNCE_CARD: {"player": 0, "opcodes": [1234, OPCODE_ISCODE]},
    # Payload is ignored; the extractor always returns rock/paper/scissors.
    MSG_ROCK_PAPER_SCISSORS: {"player": 0},
    MSG_SELECT_COUNTER: {
        "player": 0,
        "counter_type": COUNTER_NEED_ENABLE | 1,
        "count": 1,
        "cards": [{**_CARD, "counter_count": 3}],
    },
}


# Route cases: prompts whose action bytes take a route no MINIMAL_MSGS entry
# reaches. Three bytes carry a position-shaped value by different routes, and a
# per-kind case can leave any of them silently zero -- MINIMAL_MSGS' attackable
# card, for instance, always has `direct_attackable: 0`, so byte 12 never turns
# on. Keyed by case name rather than msg_type because two of them are the same
# msg_type. Shared with the golden capture script, so the bytes the tests assert
# on and the bytes the fixture freezes come from the same prompt.
ROUTE_CASES: dict[str, dict] = {
    "byte10_position_branch": {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "min": 1,
        "max": 1,
        "cards": [
            {
                "code": 100,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "subsequence": POS_FACEUP,
            }
        ],
    },
    # LOCATION_OVERLAY set, so the slot is a stack index, not a position.
    "byte10_overlay_branch": {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "min": 1,
        "max": 1,
        "cards": [
            {
                "code": 100,
                "controller": 0,
                "location": LOCATION_MZONE | LOCATION_OVERLAY,
                "sequence": 0,
                "subsequence": 2,
            }
        ],
    },
    # The chain extractor is byte 11's only producer.
    "byte11_chain_route": {
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "forced": False,
        "chains": [
            {
                "code": 200,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 1,
                "position": POS_FACEUP,
                "desc": 0,
            }
        ],
    },
    # _extract_position_actions puts its bitmask in `index`, so it lands in
    # byte 16 -- not byte 11, despite being a position.
    "byte16_choose_position_route": {
        "msg_type": MSG_SELECT_POSITION,
        "player": 0,
        "code": 300,
        "positions": POS_FACEUP_ATTACK | POS_FACEUP_DEFENSE,
    },
    "byte12_direct_attackable_route": {
        "msg_type": MSG_SELECT_BATTLECMD,
        "player": 0,
        "activatable": [],
        "attackable": [
            {
                "code": 500,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "direct_attackable": 1,
            }
        ],
    },
}


# Mid-selection prompts: a message plus the picks already made, which
# MINIMAL_MSGS cannot express. Shared with the golden capture script so the
# fixture it writes and the tests that read it cannot describe different
# prompts.
CARD_A = {"code": 111, "controller": 0, "location": LOCATION_MZONE, "sequence": 0}
CARD_B = {"code": 222, "controller": 0, "location": LOCATION_MZONE, "sequence": 1}
CARD_C = {"code": 333, "controller": 0, "location": LOCATION_MZONE, "sequence": 2}

MULTI_STEP_CASES: dict[str, tuple[dict, list[int]]] = {
    "card_mid_select": (
        {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 2,
            "cards": [CARD_A, CARD_B, CARD_C],
        },
        [1],
    ),
    "tribute_mid_select": (
        {
            "msg_type": MSG_SELECT_TRIBUTE,
            "player": 0,
            "cancelable": 0,
            "min": 2,
            "max": 2,
            "cards": [{**CARD_A, "release_param": 1}, {**CARD_B, "release_param": 1}],
        },
        [0],
    ),
    "sum_mid_select": (
        {
            "msg_type": MSG_SELECT_SUM,
            "player": 0,
            "select_type": 0,
            "target_sum": 8,
            "min": 1,
            "max": 3,
            "must_cards": [],
            "optional_cards": [
                {**CARD_A, "param": 4},
                {**CARD_B, "param": 4},
                {**CARD_C, "param": 4},
            ],
        },
        [0],
    ),
    # With card 1 picked (running sum 1), card 0 can never reach target 3
    # (1+1=2, 1+1+2=4), so it is pruned from the offered list while card 2
    # stays. The pruned card sits BELOW a picked one, which is what makes a
    # position differ from an engine index -- every other case has them
    # equal, and emitting raw engine indices for `selected` passes those.
    "sum_pruned_mid_select": (
        {
            "msg_type": MSG_SELECT_SUM,
            "player": 0,
            "select_type": 0,
            "target_sum": 3,
            "min": 1,
            "max": 3,
            "must_cards": [],
            "optional_cards": [
                {**CARD_A, "param": 1},
                {**CARD_B, "param": 1},
                {**CARD_C, "param": 2},
            ],
        },
        [1],
    ),
    "card_two_picked": (
        {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 3,
            "cards": [CARD_A, CARD_B, CARD_C],
        },
        [2, 0],
    ),
}
