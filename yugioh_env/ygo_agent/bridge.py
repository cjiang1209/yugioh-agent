"""Translate yugioh-agent observations into ygo-agent JSON API format.

Pure functions — no HTTP, no state, no ygo-agent imports. Builds
JSON-serializable dicts conforming to ygo-agent's ``ygoinf`` server schema.
"""

from __future__ import annotations

import numpy as np

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
