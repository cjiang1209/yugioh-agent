"""Translate yugioh-agent observations into ygo-agent JSON API format.

Pure functions — no HTTP, no state, no ygo-agent imports. Builds
JSON-serializable dicts conforming to ygo-agent's ``ygoinf`` server schema.
"""

from __future__ import annotations

import logging

import numpy as np

from yugioh_core.action_categories import (
    BATTLE_TO_EP,
    BATTLE_TO_M2,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from yugioh_core.constants import (
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_SELECT_BATTLECMD,
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
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_core.encoding import decode_u16, decode_u32

# ---------------------------------------------------------------------------
# Mapping tables: our bitmask/int values → ygo-agent enum string names
# ---------------------------------------------------------------------------

_LOCATION_MAP: dict[int, str] = {
    0x01: "deck",
    0x02: "hand",
    0x04: "mzone",
    0x08: "szone",
    0x10: "grave",
    0x20: "removed",
    0x40: "extra",
}

_POSITION_MAP: dict[int, str] = {
    0x0: "none",
    0x1: "faceup_attack",
    0x2: "facedown_attack",
    0x3: "attack",
    0x4: "faceup_defense",
    0x5: "faceup",
    0x6: "facedown_defense",  # not standard but defensive
    0x8: "facedown_defense",
    0xA: "facedown",
    0xC: "defense",
}

# ygopro-core attribute bitmask → ygo-agent enum name.
_ATTRIBUTE_MAP: dict[int, str] = {
    0x00: "none",
    0x01: "earth",
    0x02: "water",
    0x04: "fire",
    0x08: "wind",
    0x10: "light",
    0x20: "dark",
    0x40: "divine",
}

# Race bitmask → enum name.  Only the lowest set bit matters (cards have one race).
_RACE_MAP: dict[int, str] = {
    0x0001: "warrior",
    0x0002: "spellcaster",
    0x0004: "fairy",
    0x0008: "fiend",
    0x0010: "zombie",
    0x0020: "machine",
    0x0040: "aqua",
    0x0080: "pyro",
    0x0100: "rock",
    0x0200: "windbeast",
    0x0400: "plant",
    0x0800: "insect",
    0x1000: "thunder",
    0x2000: "dragon",
    0x4000: "beast",
    0x8000: "beast_warrior",
    0x10000: "dinosaur",
    0x20000: "fish",
    0x40000: "sea_serpent",
    0x80000: "reptile",
    0x100000: "psycho",
    0x200000: "devine",
    0x400000: "creator_god",
    0x800000: "wyrm",
    0x1000000: "cyberse",
    0x2000000: "illusion",
}

# Type bitmask bit positions → enum names (multi-hot).
_TYPE_BITS: list[tuple[int, str]] = [
    (0x1, "monster"),
    (0x2, "spell"),
    (0x4, "trap"),
    (0x10, "normal"),
    (0x20, "effect"),
    (0x40, "fusion"),
    (0x80, "ritual"),
    (0x100, "trap_monster"),
    (0x200, "spirit"),
    (0x400, "union"),
    (0x800, "dual"),
    (0x1000, "tuner"),
    (0x2000, "synchro"),
    (0x4000, "token"),
    (0x10000, "quick_play"),
    (0x20000, "continuous"),
    (0x40000, "equip"),
    (0x80000, "field"),
    (0x100000, "counter"),
    (0x200000, "flip"),
    (0x400000, "toon"),
    (0x800000, "xyz"),
    (0x1000000, "pendulum"),
    (0x2000000, "special"),
    (0x4000000, "link"),
]

# Phase bitmask → enum name.
_PHASE_MAP: dict[int, str] = {
    0x01: "draw",
    0x02: "standby",
    0x04: "main1",
    0x08: "battle_start",
    0x10: "battle_step",
    0x20: "damage",
    0x40: "damage_calculation",
    0x80: "battle",
    0x100: "main2",
    0x200: "end",
}


def _decode_location(loc_byte: int) -> str:
    return _LOCATION_MAP.get(loc_byte, "deck")


def _decode_position(pos_byte: int) -> str:
    return _POSITION_MAP.get(pos_byte, "none")


def _decode_attribute(attr_byte: int) -> str:
    return _ATTRIBUTE_MAP.get(attr_byte, "none")


def _decode_race(race_u32: int) -> str:
    return _RACE_MAP.get(race_u32, "none")


def _decode_types(type_u32: int) -> list[str]:
    return [name for bit, name in _TYPE_BITS if type_u32 & bit]


def _decode_phase(phase_u16: int) -> str:
    return _PHASE_MAP.get(phase_u16, "main1")


# ---------------------------------------------------------------------------
# Card translation
# ---------------------------------------------------------------------------


def translate_cards(obs_cards: np.ndarray) -> list[dict]:
    """Convert obs card array (MAX_CARDS × CARD_FEATURES, uint8) to ygo-agent Card dicts.

    Skips empty slots (location byte == 0 and code == 0).
    """
    cards: list[dict] = []
    for i in range(obs_cards.shape[0]):
        row = obs_cards[i]
        code = decode_u32(row, 0)
        loc_byte = int(row[4])
        # Skip empty slots
        if code == 0 and loc_byte == 0:
            continue

        type_u32 = decode_u32(row, 9)
        race_u32 = decode_u32(row, 15)
        atk = decode_u16(row, 19)
        defense = decode_u16(row, 21)
        is_overlay = bool(row[29])

        cards.append(
            {
                "code": code,
                "location": _decode_location(loc_byte),
                "sequence": int(row[5]),
                "controller": "me" if row[7] == 0 else "opponent",
                "position": _decode_position(int(row[6])),
                "overlay_sequence": 0 if is_overlay else -1,
                "attribute": _decode_attribute(int(row[14])),
                "race": _decode_race(race_u32),
                "level": int(row[13]),
                "counter": int(row[27]),
                "negated": bool(row[28]),
                "attack": atk,
                "defense": defense,
                "types": _decode_types(type_u32),
            }
        )
    return cards


# ---------------------------------------------------------------------------
# Global translation
# ---------------------------------------------------------------------------


def translate_global(obs_global: np.ndarray) -> dict:
    """Convert obs global_state array (20 uint8) to ygo-agent Global dict."""
    my_lp = decode_u16(obs_global, 0)
    opp_lp = decode_u16(obs_global, 2)
    turn = int(obs_global[4])
    phase = decode_u16(obs_global, 5)
    is_my_turn = bool(obs_global[7])
    # is_first: odd turns belong to the first player
    is_first = (turn % 2 == 1) == is_my_turn

    return {
        "my_lp": my_lp,
        "op_lp": opp_lp,
        "turn": turn,
        "phase": _decode_phase(phase),
        "is_first": is_first,
        "is_my_turn": is_my_turn,
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


def _card_info(card: dict) -> dict:
    """Build a ygo-agent CardInfo dict from our msg card dict."""
    return {
        "code": card.get("code", 0),
        "controller": "me" if card.get("controller", 0) == 0 else "opponent",
        "location": _decode_location(card.get("location", 0)),
        "sequence": card.get("sequence", 0),
    }


def _card_location(card: dict) -> dict:
    """Build a ygo-agent CardLocation dict from our msg card dict."""
    return {
        "controller": "me" if card.get("controller", 0) == 0 else "opponent",
        "location": _decode_location(card.get("location", 0)),
        "sequence": card.get("sequence", 0),
        "overlay_sequence": card.get("subsequence", -1),
    }


def _translate_idle_cmd(msg: dict) -> dict:
    """MSG_SELECT_IDLECMD → select_idlecmd."""
    cmds = []

    for category, key, cmd_type in [
        (0, "summonable", "summon"),
        (1, "sp_summonable", "sp_summon"),
        (2, "repositionable", "reposition"),
        (3, "mset", "mset"),
        (4, "sset", "set"),
    ]:
        for i, card in enumerate(msg.get(key, [])):
            cmds.append(
                {
                    "cmd_type": cmd_type,
                    "data": {
                        "card_info": _card_info(card),
                        "effect_description": 0,
                        "response": (i << 16) | category,
                    },
                }
            )

    for i, card in enumerate(msg.get("activatable", [])):
        desc = card.get("desc", 0)
        cmds.append(
            {
                "cmd_type": "activate",
                "data": {
                    "card_info": _card_info(card),
                    "effect_description": _convert_desc(int(desc)),
                    "response": (i << 16) | 5,
                },
            }
        )

    if msg.get("to_bp"):
        cmds.append({"cmd_type": "to_bp", "data": None})
    if msg.get("to_ep"):
        cmds.append({"cmd_type": "to_ep", "data": None})

    return {"data": {"msg_type": "select_idlecmd", "idle_cmds": cmds}}


def _translate_chain(msg: dict) -> dict:
    """MSG_SELECT_CHAIN → select_chain."""
    chains = []
    for i, chain in enumerate(msg.get("chains", [])):
        chains.append(
            {
                "code": chain.get("code", 0),
                "location": _card_location(chain),
                "effect_description": _convert_desc(int(chain.get("desc", 0))),
                "response": i,
            }
        )
    return {
        "data": {
            "msg_type": "select_chain",
            "forced": bool(msg.get("forced", 0)),
            "chains": chains,
        }
    }


def _translate_battlecmd(msg: dict) -> dict:
    """MSG_SELECT_BATTLECMD → select_battlecmd."""
    cmds = []
    for i, card in enumerate(msg.get("activatable", [])):
        desc = card.get("desc", 0)
        cmds.append(
            {
                "cmd_type": "activate",
                "data": {
                    "card_info": _card_info(card),
                    "effect_description": _convert_desc(int(desc)),
                    "direct_attackable": False,
                    "response": (i << 16) | 0,
                },
            }
        )
    for i, card in enumerate(msg.get("attackable", [])):
        cmds.append(
            {
                "cmd_type": "attack",
                "data": {
                    "card_info": _card_info(card),
                    "effect_description": 0,
                    "direct_attackable": bool(card.get("direct_attackable", 0)),
                    "response": (i << 16) | 1,
                },
            }
        )
    if msg.get("to_m2"):
        cmds.append({"cmd_type": "to_m2", "data": None})
    if msg.get("to_ep"):
        cmds.append({"cmd_type": "to_ep", "data": None})
    return {"data": {"msg_type": "select_battlecmd", "battle_cmds": cmds}}


def _translate_effectyn(msg: dict) -> dict:
    """MSG_SELECT_EFFECTYN → select_effectyn."""
    return {
        "data": {
            "msg_type": "select_effectyn",
            "code": msg.get("code", 0),
            "location": _card_location(msg),
            "effect_description": _convert_desc(int(msg.get("desc", 0))),
        }
    }


def _translate_yesno(msg: dict) -> dict:
    """MSG_SELECT_YESNO → select_yesno."""
    return {
        "data": {
            "msg_type": "select_yesno",
            "effect_description": _convert_desc(int(msg.get("desc", 0))),
        }
    }


def _translate_option(msg: dict) -> dict:
    """MSG_SELECT_OPTION → select_option."""
    options = [
        {"code": _convert_desc(int(o)), "response": i} for i, o in enumerate(msg.get("options", []))
    ]
    return {"data": {"msg_type": "select_option", "options": options}}


def _translate_card(msg: dict) -> dict:
    """MSG_SELECT_CARD → select_card."""
    cards = []
    for i, card in enumerate(msg.get("cards", [])):
        cards.append(
            {
                "location": _card_location(card),
                "response": i,
            }
        )
    return {
        "data": {
            "msg_type": "select_card",
            "cancelable": bool(msg.get("cancelable", 0)),
            "min": msg.get("min", 1),
            "max": msg.get("max", 1),
            "cards": cards,
            "selected": list(msg.get("_selected", [])),
        }
    }


def _translate_position(msg: dict) -> dict:
    """MSG_SELECT_POSITION → select_position."""
    positions_bitmask = msg.get("positions", 0)
    positions = []
    for pos_val, pos_name in [
        (POS_FACEUP_ATTACK, "faceup_attack"),
        (POS_FACEDOWN_ATTACK, "facedown_attack"),
        (POS_FACEUP_DEFENSE, "faceup_defense"),
        (POS_FACEDOWN_DEFENSE, "facedown_defense"),
    ]:
        if positions_bitmask & pos_val:
            positions.append(pos_name)
    return {
        "data": {
            "msg_type": "select_position",
            "code": msg.get("code", 0),
            "positions": positions,
        }
    }


def _translate_place(msg: dict, msg_type_name: str) -> dict:
    """MSG_SELECT_PLACE / MSG_SELECT_DISFIELD → select_place / select_disfield."""
    field_mask = msg.get("field_mask", 0)
    places = []
    for rel_player in range(2):
        base_m = rel_player * 16
        base_s = rel_player * 16 + 8
        controller = "me" if rel_player == 0 else "opponent"
        for seq in range(7):
            bit = base_m + seq
            if bit < 32 and not (field_mask & (1 << bit)):
                places.append(
                    {
                        "controller": controller,
                        "location": "mzone",
                        "sequence": seq,
                    }
                )
        for seq in range(6):
            bit = base_s + seq
            if bit < 32 and not (field_mask & (1 << bit)):
                places.append(
                    {
                        "controller": controller,
                        "location": "szone",
                        "sequence": seq,
                    }
                )
    return {
        "data": {
            "msg_type": msg_type_name,
            "count": msg.get("count", 1),
            "places": places,
        }
    }


def _translate_tribute(msg: dict) -> dict:
    """MSG_SELECT_TRIBUTE → select_tribute."""
    cards = []
    for i, card in enumerate(msg.get("cards", [])):
        cards.append(
            {
                "location": _card_location(card),
                "level": int(card.get("release_param", 1)),
                "response": i,
            }
        )
    return {
        "data": {
            "msg_type": "select_tribute",
            "cancelable": bool(msg.get("cancelable", 0)),
            "min": msg.get("min", 1),
            "max": msg.get("max", 1),
            "cards": cards,
            "selected": list(msg.get("_selected", [])),
        }
    }


def _translate_sum(msg: dict) -> dict:
    """MSG_SELECT_SUM → select_sum."""
    must_cards = []
    for card in msg.get("must_cards", []):
        param = card.get("param", 0)
        must_cards.append(
            {
                "location": _card_location(card),
                "level1": param & 0xFFFF,
                "level2": (param >> 16) & 0xFFFF,
                "response": -1,
            }
        )
    opt_cards = []
    for i, card in enumerate(msg.get("optional_cards", [])):
        param = card.get("param", 0)
        opt_cards.append(
            {
                "location": _card_location(card),
                "level1": param & 0xFFFF,
                "level2": (param >> 16) & 0xFFFF,
                "response": i,
            }
        )
    return {
        "data": {
            "msg_type": "select_sum",
            "overflow": bool(msg.get("select_type", 0)),
            "level_sum": msg.get("target_sum", 0),
            "min": msg.get("min", 1),
            "max": msg.get("max", 0),
            "must_cards": must_cards,
            "cards": opt_cards,
            "selected": list(msg.get("_selected", [])),
        }
    }


def _translate_unselect_card(msg: dict) -> dict:
    """MSG_SELECT_UNSELECT_CARD → select_unselect_card."""
    selectable = []
    for i, card in enumerate(msg.get("selectable", [])):
        selectable.append(
            {
                "location": _card_location(card),
                "response": i,
            }
        )
    return {
        "data": {
            "msg_type": "select_unselect_card",
            "finishable": bool(msg.get("finishable", 0)),
            "cancelable": bool(msg.get("cancelable", 0)),
            "min": msg.get("min", 1),
            "max": msg.get("max", 1),
            "selected_cards": [],
            "selectable_cards": selectable,
        }
    }


def _translate_announce_attrib(msg: dict) -> dict:
    """MSG_ANNOUNCE_ATTRIB → announce_attrib."""
    available = msg.get("available", 0)
    attribs = []
    for bit in range(8):
        mask = 1 << bit
        if available & mask:
            attribs.append(
                {
                    "attribute": _decode_attribute(mask),
                    "response": mask,
                }
            )
    return {
        "data": {
            "msg_type": "announce_attrib",
            "count": msg.get("count", 1),
            "attributes": attribs,
        }
    }


def _translate_announce_number(msg: dict) -> dict:
    """MSG_ANNOUNCE_NUMBER → announce_number."""
    numbers = []
    for i, num in enumerate(msg.get("numbers", [])):
        numbers.append({"number": int(num), "response": i})
    return {
        "data": {
            "msg_type": "announce_number",
            "count": msg.get("count", 1),
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
    MSG_SELECT_PLACE: lambda msg: _translate_place(msg, "select_place"),
    MSG_SELECT_DISFIELD: lambda msg: _translate_place(msg, "select_disfield"),
    MSG_SELECT_TRIBUTE: _translate_tribute,
    MSG_SELECT_SUM: _translate_sum,
    MSG_SELECT_UNSELECT_CARD: _translate_unselect_card,
    MSG_ANNOUNCE_ATTRIB: _translate_announce_attrib,
    MSG_ANNOUNCE_NUMBER: _translate_announce_number,
}


def translate_action_msg(msg: dict) -> dict:
    """Convert our msg dict to a ygo-agent ActionMsg JSON dict.

    Returns a dict with a ``data`` key containing the msg-type-specific payload.
    Raises ``ValueError`` for unsupported message types.
    """
    msg_type = msg.get("msg_type", 0)
    translator = _ACTION_MSG_TRANSLATORS.get(msg_type)
    if translator is None:
        raise ValueError(f"Unsupported msg_type for ygo-agent bridge: {msg_type}")
    return translator(msg)


# ---------------------------------------------------------------------------
# Top-level predict input assembly
# ---------------------------------------------------------------------------


def build_predict_input(
    obs: dict[str, np.ndarray],
    msg: dict,
    prev_action_idx: int,
    index: int = 0,
) -> dict:
    """Build the full DuelPredictRequest body for the ygo-agent server.

    Args:
        obs: Observation dict with ``cards`` and ``global_state`` arrays.
        msg: The current SELECT message dict.
        prev_action_idx: Index of the previously selected action (0 for first).
        index: Duel session index (must match server state).

    Returns:
        JSON-serializable dict matching ``DuelPredictRequest`` schema.
    """
    cards = translate_cards(obs["cards"])
    global_state = translate_global(obs["global_state"])
    action_msg = translate_action_msg(msg)

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


def _match_cmd_response(
    actions: list[dict],
    response: int,
    phase_a: int,
    phase_b: int,
) -> int:
    """Match idle/battle cmd response. Phase responses use raw category values;
    card responses use ``(index << 16) | category`` encoding."""
    if response == phase_a:
        for i, a in enumerate(actions):
            if a.get("category") == phase_a:
                return i
    elif response == phase_b:
        for i, a in enumerate(actions):
            if a.get("category") == phase_b:
                return i
    else:
        r_cat = response & 0xFFFF
        r_idx = (response >> 16) & 0xFFFF
        for i, a in enumerate(actions):
            if a.get("category") == r_cat and a.get("index") == r_idx:
                return i
    return 0


def _match_idle_response(actions: list[dict], response: int) -> int:
    return _match_cmd_response(actions, response, IDLE_TO_BP, IDLE_TO_EP)


def _match_battle_response(actions: list[dict], response: int) -> int:
    return _match_cmd_response(actions, response, BATTLE_TO_M2, BATTLE_TO_EP)


def _match_cancel_or_index(actions: list[dict], response: int) -> int:
    """Match response = item index or -1 for cancel/finish."""
    if response == -1:
        for i, a in enumerate(actions):
            if a.get("category") == 1:
                return i
    else:
        for i, a in enumerate(actions):
            if a.get("category") == 0 and a.get("index") == response:
                return i
    return 0


def _match_yesno_response(actions: list[dict], response: int) -> int:
    """Match yes/no response. 1=yes(category 0), 0=no(category 1)."""
    target_cat = 0 if response == 1 else 1
    for i, a in enumerate(actions):
        if a.get("category") == target_cat:
            return i
    return 0


def _match_position_response(actions: list[dict], response: int) -> int:
    """Match position response. response = position bitmask."""
    for i, a in enumerate(actions):
        if a.get("index") == response:
            return i
    return 0


def _match_index_response(actions: list[dict], response: int) -> int:
    """Match by action index (select_card, select_tribute, select_sum, select_option, announce_number)."""
    for i, a in enumerate(actions):
        if a.get("category") == 0 and a.get("index") == response:
            return i
    return 0


def _match_unselect_response(actions: list[dict], response: int) -> int:
    return _match_cancel_or_index(actions, response)


def _match_place_response(actions: list[dict], response: int) -> int:
    """Match place/disfield response by index position."""
    if 0 <= response < len(actions):
        return response
    return 0


def _match_attrib_response(actions: list[dict], response: int) -> int:
    """Match announce_attrib response. response = attribute bitmask."""
    # Our actions have index = bit position; response = bitmask
    for i, a in enumerate(actions):
        if a.get("index") is not None and (1 << a["index"]) == response:
            return i
    return 0


_RESPONSE_MATCHERS: dict[int, callable] = {
    MSG_SELECT_IDLECMD: _match_idle_response,
    MSG_SELECT_BATTLECMD: _match_battle_response,
    MSG_SELECT_CHAIN: _match_cancel_or_index,
    MSG_SELECT_EFFECTYN: _match_yesno_response,
    MSG_SELECT_YESNO: _match_yesno_response,
    MSG_SELECT_POSITION: _match_position_response,
    MSG_SELECT_CARD: _match_index_response,
    MSG_SELECT_TRIBUTE: _match_index_response,
    MSG_SELECT_SUM: _match_index_response,
    MSG_SELECT_OPTION: _match_index_response,
    MSG_SELECT_UNSELECT_CARD: _match_unselect_response,
    MSG_SELECT_PLACE: _match_place_response,
    MSG_SELECT_DISFIELD: _match_place_response,
    MSG_ANNOUNCE_ATTRIB: _match_attrib_response,
    MSG_ANNOUNCE_NUMBER: _match_index_response,
}


def match_response(msg_type: int, actions: list[dict], response: int) -> int:
    """Map a ygo-agent server response value to our action index.

    Args:
        msg_type: Our MSG_SELECT_* constant.
        actions: Our ActionMapper.actions list.
        response: The ``response`` field from ygo-agent's ``ActionPredict``.

    Returns:
        Action index in ``[0, len(actions))``. Falls back to 0 on no match.
    """
    matcher = _RESPONSE_MATCHERS.get(msg_type)
    if matcher is None:
        logger.warning("No response matcher for msg_type=%d, defaulting to action 0", msg_type)
        return 0
    return matcher(actions, response)
