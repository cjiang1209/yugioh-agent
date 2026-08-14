"""Translate yugioh-agent observations into ygo-agent JSON API format.

Pure functions — no HTTP, no state, no ygo-agent imports. Builds
JSON-serializable dicts conforming to ygo-agent's ``ygoinf`` server schema.

Sourced off ``YuGiOhObservation.action_descriptors`` / ``prompt_meta``
(structured, engine-agnostic) rather than the raw ygopro-core ``msg`` dict.
"Tier 3" prompts (place_zone / choose_position / pick_bit) are *derived*,
not read: the descriptor list itself IS the legal-options list, since the
mapper that builds it already applied whatever engine-side filtering
(field_mask, positions bitmask, etc.) would otherwise need re-deriving here.
"""

from __future__ import annotations

import logging

from yugioh_core.action_categories import (
    BATTLE_ACTIVATE,
    BATTLE_ATTACK,
    BATTLE_TO_EP,
    BATTLE_TO_M2,
    IDLE_ACTIVATE,
    IDLE_MSET,
    IDLE_REPOSITION,
    IDLE_SP_SUMMON,
    IDLE_SSET,
    IDLE_SUMMON,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from yugioh_core.constants import (
    ATTRIBUTE_DARK,
    ATTRIBUTE_DIVINE,
    ATTRIBUTE_EARTH,
    ATTRIBUTE_FIRE,
    ATTRIBUTE_LIGHT,
    ATTRIBUTE_WATER,
    ATTRIBUTE_WIND,
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    PHASE_BATTLE,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_DRAW,
    PHASE_END,
    PHASE_MAIN1,
    PHASE_MAIN2,
    PHASE_STANDBY,
    POS_ATTACK,
    POS_DEFENSE,
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    RACE_AQUA,
    RACE_BEAST,
    RACE_BEASTWARRIOR,
    RACE_CREATORGOD,
    RACE_CYBERSE,
    RACE_DINOSAUR,
    RACE_DIVINE,
    RACE_DRAGON,
    RACE_FAIRY,
    RACE_FIEND,
    RACE_FISH,
    RACE_ILLUSION,
    RACE_INSECT,
    RACE_MACHINE,
    RACE_PLANT,
    RACE_PSYCHIC,
    RACE_PYRO,
    RACE_REPTILE,
    RACE_ROCK,
    RACE_SEASERPENT,
    RACE_SPELLCASTER,
    RACE_THUNDER,
    RACE_WARRIOR,
    RACE_WINGEDBEAST,
    RACE_WYRM,
    RACE_ZOMBIE,
    TYPE_CONTINUOUS,
    TYPE_COUNTER,
    TYPE_EFFECT,
    TYPE_EQUIP,
    TYPE_FIELD,
    TYPE_FLIP,
    TYPE_FUSION,
    TYPE_GEMINI,
    TYPE_LINK,
    TYPE_MONSTER,
    TYPE_NORMAL,
    TYPE_PENDULUM,
    TYPE_QUICKPLAY,
    TYPE_RITUAL,
    TYPE_SPELL,
    TYPE_SPIRIT,
    TYPE_SPSUMMON,
    TYPE_SYNCHRO,
    TYPE_TOKEN,
    TYPE_TOON,
    TYPE_TRAP,
    TYPE_TRAPMONSTER,
    TYPE_TUNER,
    TYPE_UNION,
    TYPE_XYZ,
)
from yugioh_env.models import (
    ActivateEffect,
    Attack,
    CardCommand,
    CardRef,
    ChooseOption,
    ChoosePosition,
    Confirm,
    FinishPick,
    Pass,
    PhaseChange,
    PickBit,
    PickCard,
    PlaceZone,
    YuGiOhObservation,
)
from yugioh_env.models import AnnounceNumber as AnnounceNumberDescriptor

# ---------------------------------------------------------------------------
# Mapping tables: our bitmask/int values → ygo-agent enum string names
# ---------------------------------------------------------------------------

_LOCATION_MAP: dict[int, str] = {
    LOCATION_DECK: "deck",
    LOCATION_HAND: "hand",
    LOCATION_MZONE: "mzone",
    LOCATION_SZONE: "szone",
    LOCATION_GRAVE: "grave",
    LOCATION_BANISHED: "removed",
    LOCATION_EXTRA: "extra",
}

_POSITION_MAP: dict[int, str] = {
    0x0: "none",  # no POS_* constant: absent/unknown position
    POS_FACEUP_ATTACK: "faceup_attack",
    POS_FACEDOWN_ATTACK: "facedown_attack",
    POS_ATTACK: "attack",
    POS_FACEUP_DEFENSE: "faceup_defense",
    POS_FACEUP: "faceup",
    0x6: "facedown_defense",  # no POS_* constant: not standard but defensive
    POS_FACEDOWN_DEFENSE: "facedown_defense",
    POS_FACEDOWN: "facedown",
    POS_DEFENSE: "defense",
}

# ygopro-core attribute bitmask → ygo-agent enum name.
_ATTRIBUTE_MAP: dict[int, str] = {
    0x00: "none",  # no ATTRIBUTE_* constant: no attribute
    ATTRIBUTE_EARTH: "earth",
    ATTRIBUTE_WATER: "water",
    ATTRIBUTE_FIRE: "fire",
    ATTRIBUTE_WIND: "wind",
    ATTRIBUTE_LIGHT: "light",
    ATTRIBUTE_DARK: "dark",
    ATTRIBUTE_DIVINE: "divine",
}

# Race bitmask → enum name.  Only the lowest set bit matters (cards have one race).
_RACE_MAP: dict[int, str] = {
    RACE_WARRIOR: "warrior",
    RACE_SPELLCASTER: "spellcaster",
    RACE_FAIRY: "fairy",
    RACE_FIEND: "fiend",
    RACE_ZOMBIE: "zombie",
    RACE_MACHINE: "machine",
    RACE_AQUA: "aqua",
    RACE_PYRO: "pyro",
    RACE_ROCK: "rock",
    RACE_WINGEDBEAST: "windbeast",
    RACE_PLANT: "plant",
    RACE_INSECT: "insect",
    RACE_THUNDER: "thunder",
    RACE_DRAGON: "dragon",
    RACE_BEAST: "beast",
    RACE_BEASTWARRIOR: "beast_warrior",
    RACE_DINOSAUR: "dinosaur",
    RACE_FISH: "fish",
    RACE_SEASERPENT: "sea_serpent",
    RACE_REPTILE: "reptile",
    RACE_PSYCHIC: "psycho",
    RACE_DIVINE: "devine",
    RACE_CREATORGOD: "creator_god",
    RACE_WYRM: "wyrm",
    RACE_CYBERSE: "cyberse",
    RACE_ILLUSION: "illusion",
}

# Type bitmask bit positions → enum names (multi-hot).
_TYPE_BITS: list[tuple[int, str]] = [
    (TYPE_MONSTER, "monster"),
    (TYPE_SPELL, "spell"),
    (TYPE_TRAP, "trap"),
    (TYPE_NORMAL, "normal"),
    (TYPE_EFFECT, "effect"),
    (TYPE_FUSION, "fusion"),
    (TYPE_RITUAL, "ritual"),
    (TYPE_TRAPMONSTER, "trap_monster"),
    (TYPE_SPIRIT, "spirit"),
    (TYPE_UNION, "union"),
    (TYPE_GEMINI, "dual"),
    (TYPE_TUNER, "tuner"),
    (TYPE_SYNCHRO, "synchro"),
    (TYPE_TOKEN, "token"),
    (TYPE_QUICKPLAY, "quick_play"),
    (TYPE_CONTINUOUS, "continuous"),
    (TYPE_EQUIP, "equip"),
    (TYPE_FIELD, "field"),
    (TYPE_COUNTER, "counter"),
    (TYPE_FLIP, "flip"),
    (TYPE_TOON, "toon"),
    (TYPE_XYZ, "xyz"),
    (TYPE_PENDULUM, "pendulum"),
    (TYPE_SPSUMMON, "special"),
    (TYPE_LINK, "link"),
]

# Phase bitmask → enum name.
_PHASE_MAP: dict[int, str] = {
    PHASE_DRAW: "draw",
    PHASE_STANDBY: "standby",
    PHASE_MAIN1: "main1",
    PHASE_BATTLE_START: "battle_start",
    PHASE_BATTLE_STEP: "battle_step",
    PHASE_DAMAGE: "damage",
    PHASE_DAMAGE_CAL: "damage_calculation",
    PHASE_BATTLE: "battle",
    PHASE_MAIN2: "main2",
    PHASE_END: "end",
}


def _decode_location(location: int) -> str:
    return _LOCATION_MAP.get(location, "deck")


def _decode_position(position: int) -> str:
    return _POSITION_MAP.get(position, "none")


def _decode_attribute(attribute: int) -> str:
    return _ATTRIBUTE_MAP.get(attribute, "none")


def _decode_race(race: int) -> str:
    return _RACE_MAP.get(race, "none")


def _decode_types(card_type: int) -> list[str]:
    return [name for bit, name in _TYPE_BITS if card_type & bit]


def _decode_phase(phase: int) -> str:
    return _PHASE_MAP.get(phase, "main1")


# ---------------------------------------------------------------------------
# Card translation
# ---------------------------------------------------------------------------


def translate_cards(obs: YuGiOhObservation) -> list[dict]:
    """Convert obs.card_states to ygo-agent Card dicts.

    Monster and spell zones are fixed-size, so the engine reports an empty one
    as a ``code == 0`` entry carrying only its location. ygo-agent never sees
    an empty zone natively, so those are dropped, as is an entry naming no
    zone at all. A ``code == 0`` entry elsewhere (hand/deck/extra) is a
    hidden-but-real card and is kept -- the model relies on its presence for
    location counts.
    """
    cards: list[dict] = []
    for card in obs.card_states:
        # Skip fully-empty entries and empty monster/spell zone holes.
        if card.code == 0 and card.location in (0, LOCATION_MZONE, LOCATION_SZONE):
            continue

        cards.append(
            {
                "code": card.code,
                "location": _decode_location(card.location),
                "sequence": card.sequence,
                "controller": "me" if card.controller == 0 else "opponent",
                "position": _decode_position(card.position),
                "overlay_sequence": 0 if card.is_overlay else -1,
                "attribute": _decode_attribute(card.attribute),
                "race": _decode_race(card.race),
                "level": card.level,
                "counter": card.counter_count,
                "negated": card.negated,
                # A `?` stat reaches us as -2, which the server would wrap to
                # 65534 (its feature encoder reduces mod 65536), presenting the
                # card as having the largest attack in the space. Its schema
                # accepts the negative, so nothing would reject it.
                "attack": max(0, card.attack),
                "defense": max(0, card.defense),
                "types": _decode_types(card.card_type),
            }
        )
    return cards


# ---------------------------------------------------------------------------
# Global translation
# ---------------------------------------------------------------------------


def translate_global(obs: YuGiOhObservation) -> dict:
    """Convert obs.global_ to ygo-agent Global dict."""
    gs = obs.global_
    # is_first: odd turns belong to the first player
    is_first = (gs.turn % 2 == 1) == gs.is_my_turn

    return {
        "my_lp": gs.my_lp,
        "op_lp": gs.opp_lp,
        "turn": gs.turn,
        "phase": _decode_phase(gs.phase),
        "is_first": is_first,
        "is_my_turn": gs.is_my_turn,
    }


# ---------------------------------------------------------------------------
# Action message translation
# ---------------------------------------------------------------------------


# System string ID remap: edo9300 → Fluorohydride.
# The two ygopro-core forks reassigned several system string IDs.
# When a system-string desc from our engine reaches the ygo-agent server,
# it must use the Fluorohydride ID so the server's ``system_string_to_id``
# lookup succeeds and the model sees the correct embedding.
_SYSSTRING_REMAP: dict[int, int] = {
    503: 1192,  # Banish
    504: 1191,  # Send to GY
    507: 1193,  # Return to Deck
    573: 1190,  # Add to hand
    574: 1191,  # Send to GY
    1170: 1169,  # Fusion Summon
    1171: 1168,  # Ritual Summon
    1172: 1164,  # Synchro Summon
    1173: 1165,  # Xyz Summon
    1174: 1166,  # Link Summon
}
# Fluorohydride 1167 (Tribute Summon) has no edo9300 equivalent — our engine
# never produces it, so no reverse mapping is needed.


def _convert_desc(desc: int) -> int:
    """Convert edo9300 effect desc to Fluorohydride format.

    Two conversions:

    1. **Bit layout**: edo9300 uses 64-bit ``(card_code << 4 | effect_idx) << 16``;
       Fluorohydride uses 32-bit ``card_code << 4 | effect_idx``.
       Descs ≥ 0x10000 are right-shifted by 16.

    2. **System string IDs**: descs < 0x10000 are system string references.
       Some IDs were reassigned between the edo9300 and Fluorohydride forks
       (e.g. "Fusion Summon" moved from 1170 to 1169).  These are remapped
       via ``_SYSSTRING_REMAP``.
    """
    if desc < 0x10000:
        return _SYSSTRING_REMAP.get(desc, desc)
    return desc >> 16


def _controller_str(controller: int) -> str:
    return "me" if controller == 0 else "opponent"


def _card_ref_info(card: CardRef) -> dict:
    """Build a ygo-agent CardInfo dict from a descriptor's CardRef."""
    return {
        "code": card.code,
        "controller": _controller_str(card.controller),
        "location": _decode_location(card.location),
        "sequence": card.sequence,
    }


def _zone(controller: int, location: int, sequence: int) -> dict:
    """Build a ygo-agent CardLocation dict.

    ``overlay_sequence`` is always -1: nothing that reaches here refers to an
    Xyz overlay material — these are field/hand/deck cards. A real overlay
    index produces specs like ``h1a11`` that match no card in the list,
    leaving the model unable to tell the cards apart.
    """
    return {
        "controller": _controller_str(controller),
        "location": _decode_location(location),
        "sequence": sequence,
        "overlay_sequence": -1,
    }


def _card_ref_location(card: CardRef) -> dict:
    return _zone(card.controller, card.location, card.sequence)


_IDLE_CMD_NAMES: dict[int, str] = {
    IDLE_SUMMON: "summon",
    IDLE_SP_SUMMON: "sp_summon",
    IDLE_REPOSITION: "reposition",
    IDLE_MSET: "mset",
    IDLE_SSET: "set",
}


def _translate_idle_cmd(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_IDLECMD → select_idlecmd."""
    cmds = []
    for d in descriptors:
        if isinstance(d, CardCommand):
            cmds.append(
                {
                    "cmd_type": _IDLE_CMD_NAMES[d.command],
                    "data": {
                        "card_info": _card_ref_info(d.card),
                        "effect_description": 0,
                        "response": (d.engine_index << 16) | d.command,
                    },
                }
            )
        elif isinstance(d, ActivateEffect):
            cmds.append(
                {
                    "cmd_type": "activate",
                    "data": {
                        "card_info": _card_ref_info(d.card),
                        "effect_description": _convert_desc(d.desc),
                        "response": (d.engine_index << 16) | IDLE_ACTIVATE,
                    },
                }
            )
        elif isinstance(d, PhaseChange):
            cmds.append({"cmd_type": f"to_{d.to}", "data": None})
    return {"data": {"msg_type": "select_idlecmd", "idle_cmds": cmds}}


def _translate_chain(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_CHAIN → select_chain."""
    chains = [
        {
            "code": d.card.code,
            "location": _card_ref_location(d.card),
            "effect_description": _convert_desc(d.desc),
            "response": d.engine_index,
        }
        for d in descriptors
        if isinstance(d, ActivateEffect)
    ]
    return {
        "data": {
            "msg_type": "select_chain",
            "forced": bool(prompt_meta["forced"]),
            "chains": chains,
        }
    }


def _translate_battlecmd(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_BATTLECMD → select_battlecmd."""
    cmds = []
    for d in descriptors:
        if isinstance(d, ActivateEffect):
            cmds.append(
                {
                    "cmd_type": "activate",
                    "data": {
                        "card_info": _card_ref_info(d.card),
                        "effect_description": _convert_desc(d.desc),
                        "direct_attackable": False,
                        "response": (d.engine_index << 16) | BATTLE_ACTIVATE,
                    },
                }
            )
        elif isinstance(d, Attack):
            cmds.append(
                {
                    "cmd_type": "attack",
                    "data": {
                        "card_info": _card_ref_info(d.card),
                        "effect_description": 0,
                        "direct_attackable": d.direct_attackable,
                        "response": (d.engine_index << 16) | BATTLE_ATTACK,
                    },
                }
            )
        elif isinstance(d, PhaseChange):
            cmds.append({"cmd_type": f"to_{d.to}", "data": None})
    return {"data": {"msg_type": "select_battlecmd", "battle_cmds": cmds}}


def _translate_effectyn(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_EFFECTYN → select_effectyn.

    No descriptor carries the prompt's card (Confirm is fieldless besides
    yes/desc), so the card ref is read entirely from prompt_meta.
    """
    return {
        "data": {
            "msg_type": "select_effectyn",
            "code": prompt_meta["card_code"],
            "location": _zone(
                prompt_meta["controller"], prompt_meta["location"], prompt_meta["sequence"]
            ),
            "effect_description": _convert_desc(int(prompt_meta["desc"])),
        }
    }


def _translate_yesno(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_YESNO → select_yesno."""
    return {
        "data": {
            "msg_type": "select_yesno",
            "effect_description": _convert_desc(int(prompt_meta["desc"])),
        }
    }


def _translate_option(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_OPTION → select_option."""
    options = [
        {"code": _convert_desc(d.desc), "response": d.engine_index}
        for d in descriptors
        if isinstance(d, ChooseOption)
    ]
    return {"data": {"msg_type": "select_option", "options": options}}


def _picked_card_location(c: dict) -> dict:
    return _zone(c["controller"], c["location"], c["sequence"])


def _reconstruct_pick_list(
    descriptors: list, prompt_meta: dict, render_new, render_picked
) -> tuple[list, list[int]]:
    """Merge the (already-compacted) descriptor list with already-picked
    cards from ``prompt_meta`` into one engine-index-ordered list, and report
    where each picked card landed in it.

    ``_extract_multi_step_actions`` filters already-picked entries out of the
    descriptor list (to avoid re-offering them as choices) and, for
    ``MSG_SELECT_SUM``, additionally prunes optional cards that the
    ``reachable`` guard determined cannot participate in any valid
    completion (see ``_extract_sum_actions`` in ``action_space.py``). Either
    filter can leave gaps in the engine-index space, so the merged list is
    NOT assumed contiguous: it is simply every remaining engine index in
    ascending order.

    The server reads each entry's own ``response`` as an engine index, but
    reads the prompt's ``selected`` positionally, as indices into the list we
    send. So ``response`` carries the true engine index while ``selected``
    must be translated into positions — the second element returned here.
    The two coincide whenever nothing is pruned or truncated, which is what
    makes confusing them easy to miss.

    Args:
        render_new: ``(descriptor) -> dict``, renders a not-yet-picked entry.
        render_picked: ``(picked_card_dict, engine_index) -> dict``, renders
            an already-picked entry from ``prompt_meta["picked_cards"]``.

    Returns:
        ``(cards, selected_positions)`` — ``cards`` is the merged list
        ordered by ascending engine index; ``selected_positions`` gives, for
        each already-picked card (in ``prompt_meta["selected"]`` order), its
        position within ``cards``.
    """
    selected = list(prompt_meta["selected"])
    picked_cards = prompt_meta["picked_cards"]
    by_index: dict[int, dict] = {}
    for d in descriptors:
        if isinstance(d, PickCard):
            by_index[d.engine_index] = render_new(d)
    # `selected` and `picked_cards` are built from the same iteration in
    # _build_prompt_meta, so they pair up positionally. strict=True: a length
    # mismatch means the producer changed and every pairing below is suspect.
    for engine_index, card in zip(selected, picked_cards, strict=True):
        by_index[engine_index] = render_picked(card, engine_index)
    ordered_indices = sorted(by_index)
    cards = [by_index[i] for i in ordered_indices]
    position_of = {engine_index: pos for pos, engine_index in enumerate(ordered_indices)}
    selected_positions = [position_of[engine_index] for engine_index in selected]
    return cards, selected_positions


def _translate_card(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_CARD → select_card."""
    cards, selected_positions = _reconstruct_pick_list(
        descriptors,
        prompt_meta,
        render_new=lambda d: {"location": _card_ref_location(d.card), "response": d.engine_index},
        render_picked=lambda c, engine_index: {
            "location": _picked_card_location(c),
            "response": engine_index,
        },
    )
    return {
        "data": {
            "msg_type": "select_card",
            "cancelable": bool(prompt_meta["cancelable"]),
            "min": prompt_meta["min"],
            "max": prompt_meta["max"],
            "cards": cards,
            "selected": selected_positions,
        }
    }


def _translate_position(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_POSITION → select_position."""
    positions = [_decode_position(d.position) for d in descriptors if isinstance(d, ChoosePosition)]
    return {
        "data": {
            "msg_type": "select_position",
            "code": prompt_meta["card_code"],
            "positions": positions,
        }
    }


def _translate_place(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_PLACE → select_place.

    place_zone descriptors ARE the legal-zone list already (derived by the
    mapper from the engine's field_mask) — no bitmask re-derivation needed
    here. Places carry no explicit ``response`` field: the schema uses each
    entry's ordinal position in this list as its response value.
    """
    places = [
        {
            "controller": _controller_str(d.controller),
            "location": _decode_location(d.location),
            "sequence": d.sequence,
        }
        for d in descriptors
        if isinstance(d, PlaceZone)
    ]
    return {
        "data": {
            "msg_type": "select_place",
            "count": prompt_meta["count"],
            "places": places,
        }
    }


def _translate_tribute(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_TRIBUTE → select_tribute."""
    cards, selected_positions = _reconstruct_pick_list(
        descriptors,
        prompt_meta,
        render_new=lambda d: {
            "location": _card_ref_location(d.card),
            "level": d.param,
            "response": d.engine_index,
        },
        render_picked=lambda c, engine_index: {
            "location": _picked_card_location(c),
            "level": c["param"],
            "response": engine_index,
        },
    )
    return {
        "data": {
            "msg_type": "select_tribute",
            "cancelable": bool(prompt_meta["cancelable"]),
            "min": prompt_meta["min_release"],
            "max": prompt_meta["max_cards"],
            "cards": cards,
            "selected": selected_positions,
        }
    }


def _translate_sum(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_SUM → select_sum.

    ``must_cards`` have no descriptor of their own (the mapper never turns
    them into actions — they're mandatory, not chosen) so they're read
    entirely from prompt_meta's full must_cards dicts.

    Deliberate divergence from the pre-refactor bridge: ``_extract_sum_actions``
    (action_space.py) applies a ``reachable`` filter that prunes optional
    cards which cannot participate in any valid completion of the sum. Those
    pruned cards never get a ``PickCard`` descriptor and are not picked
    either, so they are absent from ``opt_cards`` entirely — whereas the
    pre-refactor bridge read ``msg["optional_cards"]`` directly and offered
    every optional card, reachable or not. This means a pruned sum prompt's
    body is NOT byte-equal to what the pre-refactor bridge would have sent.
    That's intentional: we stop offering the model cards our own harness
    would reject as an invalid pick. Do not re-add pruned cards to force
    byte-equality.
    """
    must_cards = [
        {
            "location": _picked_card_location(c),
            "level1": c["param"] & 0xFFFF,
            "level2": (c["param"] >> 16) & 0xFFFF,
            "response": -1,
        }
        for c in prompt_meta["must_cards"]
    ]
    opt_cards, selected_positions = _reconstruct_pick_list(
        descriptors,
        prompt_meta,
        render_new=lambda d: {
            "location": _card_ref_location(d.card),
            "level1": d.param & 0xFFFF,
            "level2": (d.param >> 16) & 0xFFFF,
            "response": d.engine_index,
        },
        render_picked=lambda c, engine_index: {
            "location": _picked_card_location(c),
            "level1": c["param"] & 0xFFFF,
            "level2": (c["param"] >> 16) & 0xFFFF,
            "response": engine_index,
        },
    )
    return {
        "data": {
            "msg_type": "select_sum",
            "overflow": bool(prompt_meta["select_type"]),
            "level_sum": prompt_meta["target_sum"],
            "min": prompt_meta["min"],
            "max": prompt_meta["max"],
            "must_cards": must_cards,
            "cards": opt_cards,
            "selected": selected_positions,
        }
    }


def _translate_unselect_card(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_SELECT_UNSELECT_CARD → select_unselect_card."""
    selectable = [
        {"location": _card_ref_location(d.card), "response": d.engine_index}
        for d in descriptors
        if isinstance(d, PickCard)
    ]
    return {
        "data": {
            "msg_type": "select_unselect_card",
            "finishable": bool(prompt_meta["finishable"]),
            "cancelable": bool(prompt_meta["cancelable"]),
            "min": prompt_meta["min"],
            "max": prompt_meta["max"],
            "selected_cards": [],
            "selectable_cards": selectable,
        }
    }


def _translate_announce_attrib(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_ANNOUNCE_ATTRIB → announce_attrib."""
    attribs = [
        {"attribute": _decode_attribute(d.value), "response": d.value}
        for d in descriptors
        if isinstance(d, PickBit)
    ]
    return {
        "data": {
            "msg_type": "announce_attrib",
            "count": prompt_meta["count"],
            "attributes": attribs,
        }
    }


def _translate_announce_number(descriptors: list, prompt_meta: dict) -> dict:
    """MSG_ANNOUNCE_NUMBER → announce_number."""
    numbers = [
        {"number": d.value, "response": d.engine_index}
        for d in descriptors
        if isinstance(d, AnnounceNumberDescriptor)
    ]
    return {
        "data": {
            "msg_type": "announce_number",
            "count": 1,  # how many numbers to announce; the server rejects anything else
            "numbers": numbers,
        }
    }


_ACTION_MSG_TRANSLATORS: dict[int, callable] = {
    MSG_SELECT_IDLECMD: _translate_idle_cmd,
    MSG_SELECT_CHAIN: _translate_chain,
    MSG_SELECT_BATTLECMD: _translate_battlecmd,
    MSG_SELECT_EFFECTYN: _translate_effectyn,
    MSG_SELECT_YESNO: _translate_yesno,
    MSG_SELECT_OPTION: _translate_option,
    MSG_SELECT_CARD: _translate_card,
    MSG_SELECT_POSITION: _translate_position,
    MSG_SELECT_PLACE: _translate_place,
    MSG_SELECT_TRIBUTE: _translate_tribute,
    MSG_SELECT_SUM: _translate_sum,
    MSG_SELECT_UNSELECT_CARD: _translate_unselect_card,
    MSG_ANNOUNCE_ATTRIB: _translate_announce_attrib,
    MSG_ANNOUNCE_NUMBER: _translate_announce_number,
}


def translate_action_msg(descriptors: list, prompt_meta: dict | None) -> dict:
    """Convert action_descriptors/prompt_meta to a ygo-agent ActionMsg JSON dict.

    Returns a dict with a ``data`` key containing the msg-type-specific payload.
    Raises ``ValueError`` for a missing prompt or an unsupported message type.
    """
    if not prompt_meta:
        raise ValueError("No active prompt: prompt_meta is missing")
    msg_type = prompt_meta.get("msg_type")
    translator = _ACTION_MSG_TRANSLATORS.get(msg_type)
    if translator is None:
        raise ValueError(f"Unsupported msg_type for ygo-agent bridge: {msg_type}")
    return translator(descriptors, prompt_meta)


# ---------------------------------------------------------------------------
# Top-level predict input assembly
# ---------------------------------------------------------------------------


def _hidden_deck_card(controller: str) -> dict:
    """A face-down, unknown deck card. Deck cards are always hidden, so only
    location/controller carry information; every other field is the ``none``/0
    default the server uses for unknown cards."""
    return {
        "code": 0,
        "location": "deck",
        "sequence": 0,
        "controller": controller,
        "position": "faceup",
        "overlay_sequence": -1,
        "attribute": "none",
        "race": "none",
        "level": 0,
        "counter": 0,
        "negated": False,
        "attack": 0,
        "defense": 0,
        "types": [],
    }


def build_predict_input(
    obs: YuGiOhObservation,
    prev_action_idx: int,
    index: int = 0,
) -> dict:
    """Build the full DuelPredictRequest body for the ygo-agent server.

    Args:
        obs: The current observation (structured card_states/global_ fields
            plus action_descriptors/prompt_meta for the active prompt).
        prev_action_idx: Index of the previously selected action (0 for first).
        index: Duel session index (must match server state).

    Returns:
        JSON-serializable dict matching ``DuelPredictRequest`` schema.
    """
    cards = translate_cards(obs)
    global_state = translate_global(obs)
    action_msg = translate_action_msg(obs.action_descriptors, obs.prompt_meta)

    # Deck cards are hidden and never appear in obs.card_states — only their
    # count lives in obs.global_. ygo-agent derives deck counts from the card
    # list, so without these placeholders the model sees an empty deck, which is
    # far off-distribution and collapses the policy toward uniform. Synthesize
    # one hidden card per deck slot to restore the true count.
    cards.extend(_hidden_deck_card("me") for _ in range(obs.global_.my_deck))
    cards.extend(_hidden_deck_card("opponent") for _ in range(obs.global_.opp_deck))

    return {
        "input": {
            "global": global_state,
            "cards": cards,
            "action_msg": action_msg,
        },
        "prev_action_idx": prev_action_idx,
        "index": index,
    }


# ---------------------------------------------------------------------------
# Response matching
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _find(descriptors: list, pred) -> int:
    """First absolute index whose descriptor is non-None and satisfies pred."""
    for i, d in enumerate(descriptors):
        if d is not None and pred(d):
            return i
    # No slot matched. Returning 0 is indistinguishable from a real match on
    # slot 0, so say so -- a wrong-but-plausible action is otherwise invisible.
    logger.warning("No descriptor matched the server response; falling back to slot 0")
    return 0


def _match_idle_response(descriptors: list, response: int) -> int:
    """Match idle cmd response. Phase responses use raw category values;
    card/activate responses use ``(engine_index << 16) | category`` encoding."""
    if response == IDLE_TO_BP:
        return _find(descriptors, lambda d: isinstance(d, PhaseChange) and d.to == "bp")
    if response == IDLE_TO_EP:
        return _find(descriptors, lambda d: isinstance(d, PhaseChange) and d.to == "ep")
    r_cat = response & 0xFFFF
    r_idx = (response >> 16) & 0xFFFF
    if r_cat == IDLE_ACTIVATE:
        return _find(
            descriptors, lambda d: isinstance(d, ActivateEffect) and d.engine_index == r_idx
        )
    return _find(
        descriptors,
        lambda d: isinstance(d, CardCommand) and d.command == r_cat and d.engine_index == r_idx,
    )


def _match_battle_response(descriptors: list, response: int) -> int:
    """Match battle cmd response. Phase responses use raw category values;
    activate/attack responses use ``(engine_index << 16) | category`` encoding."""
    if response == BATTLE_TO_M2:
        return _find(descriptors, lambda d: isinstance(d, PhaseChange) and d.to == "m2")
    if response == BATTLE_TO_EP:
        return _find(descriptors, lambda d: isinstance(d, PhaseChange) and d.to == "ep")
    r_cat = response & 0xFFFF
    r_idx = (response >> 16) & 0xFFFF
    if r_cat == BATTLE_ATTACK:
        return _find(descriptors, lambda d: isinstance(d, Attack) and d.engine_index == r_idx)
    return _find(descriptors, lambda d: isinstance(d, ActivateEffect) and d.engine_index == r_idx)


def _match_chain_response(descriptors: list, response: int) -> int:
    """Chain activate matches on the PLAIN engine_index (no tag); pass is -1."""
    if response == -1:
        return _find(descriptors, lambda d: isinstance(d, Pass))
    return _find(
        descriptors, lambda d: isinstance(d, ActivateEffect) and d.engine_index == response
    )


def _match_pick_or_finish(descriptors: list, response: int) -> int:
    """Match card/tribute/sum response = engine_index, or -1 for finish."""
    if response == -1:
        return _find(descriptors, lambda d: isinstance(d, FinishPick))
    return _find(descriptors, lambda d: isinstance(d, PickCard) and d.engine_index == response)


def _match_unselect_response(descriptors: list, response: int) -> int:
    """Match unselect response = engine_index, or -1 for pass (not finish)."""
    if response == -1:
        return _find(descriptors, lambda d: isinstance(d, Pass))
    return _find(descriptors, lambda d: isinstance(d, PickCard) and d.engine_index == response)


def _match_confirm_response(descriptors: list, response: int) -> int:
    """Match yes/no response. 1=yes, 0=no."""
    target_yes = response == 1
    return _find(descriptors, lambda d: isinstance(d, Confirm) and d.yes == target_yes)


def _match_position_response(descriptors: list, response: int) -> int:
    """Match position response. response = position bitmask."""
    return _find(descriptors, lambda d: isinstance(d, ChoosePosition) and d.position == response)


def _match_option_response(descriptors: list, response: int) -> int:
    return _find(descriptors, lambda d: isinstance(d, ChooseOption) and d.engine_index == response)


def _match_number_response(descriptors: list, response: int) -> int:
    """Match announce_number response = engine_index (NOT the announced value)."""
    return _find(
        descriptors,
        lambda d: isinstance(d, AnnounceNumberDescriptor) and d.engine_index == response,
    )


def _match_attrib_response(descriptors: list, response: int) -> int:
    """Match announce_attrib response = value (the 1<<bit MASK), NOT engine_index (the bit)."""
    return _find(descriptors, lambda d: isinstance(d, PickBit) and d.value == response)


def _match_place_response(descriptors: list, response: int) -> int:
    """Match place response = ordinal position among place_zone descriptors."""
    ordinal = 0
    for i, d in enumerate(descriptors):
        if isinstance(d, PlaceZone):
            if ordinal == response:
                return i
            ordinal += 1
    return 0


_RESPONSE_MATCHERS: dict[int, callable] = {
    MSG_SELECT_IDLECMD: _match_idle_response,
    MSG_SELECT_BATTLECMD: _match_battle_response,
    MSG_SELECT_CHAIN: _match_chain_response,
    MSG_SELECT_EFFECTYN: _match_confirm_response,
    MSG_SELECT_YESNO: _match_confirm_response,
    MSG_SELECT_POSITION: _match_position_response,
    MSG_SELECT_CARD: _match_pick_or_finish,
    MSG_SELECT_TRIBUTE: _match_pick_or_finish,
    MSG_SELECT_SUM: _match_pick_or_finish,
    MSG_SELECT_OPTION: _match_option_response,
    MSG_SELECT_UNSELECT_CARD: _match_unselect_response,
    MSG_SELECT_PLACE: _match_place_response,
    MSG_ANNOUNCE_ATTRIB: _match_attrib_response,
    MSG_ANNOUNCE_NUMBER: _match_number_response,
}


def match_response(msg_type: int, descriptors: list, response: int) -> int:
    """Map a ygo-agent server response value to our action index.

    Args:
        msg_type: Our MSG_SELECT_* constant.
        descriptors: ``YuGiOhObservation.action_descriptors`` (may contain
            ``None`` padding entries; skipped).
        response: The ``response`` field from ygo-agent's ``ActionPredict``.

    Returns:
        Action index in ``[0, len(descriptors))``. Falls back to 0 on no match.
    """
    matcher = _RESPONSE_MATCHERS.get(msg_type)
    if matcher is None:
        logger.warning("No response matcher for msg_type=%d, defaulting to action 0", msg_type)
        return 0
    return matcher(descriptors, response)
