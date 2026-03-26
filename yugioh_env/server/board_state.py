"""Build a human-readable board state dict from the live duel."""

from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

from yugioh_core.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    POS_FACEUP_ATTACK,
    POS_FACEDOWN_ATTACK,
    POS_FACEUP_DEFENSE,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEDOWN,
    POS_ATTACK,
    POS_DEFENSE,
    QUERY_BASIC,
    QUERY_END,
    TYPE_MONSTER,
    TYPE_SPELL,
    TYPE_TRAP,
    TYPE_LINK,
)
from yugioh_env.core_types import OCG_QueryInfo, c_uint32

if TYPE_CHECKING:
    from yugioh_env.duel import Duel
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

_POS_NAMES = {
    POS_FACEUP_ATTACK: "ATK",
    POS_FACEDOWN_ATTACK: "FACE_DOWN_ATK",
    POS_FACEUP_DEFENSE: "DEF",
    POS_FACEDOWN_DEFENSE: "FACE_DOWN_DEF",
    POS_FACEUP: "ATK",
    POS_FACEDOWN: "FACE_DOWN_ATK",
    POS_ATTACK: "ATK",
    POS_DEFENSE: "DEF",
}

# ─── edo9300 query buffer parser ───────────────────────────────────────────
#
# The format from OCG_DuelQueryLocation is:
#   uint32: total_data_size (bytes after this header)
#   For each slot in the location list:
#     - Empty slot: int16(0)
#     - Card: sequence of field blocks, terminated by QUERY_END
#       Each field: uint16(field_size) + uint32(flag) + data[field_size - 4]
#       Terminator: uint16(4) + uint32(QUERY_END)

_SIMPLE_U32_FLAGS = {
    0x1,      # QUERY_CODE
    0x2,      # QUERY_POSITION
    0x4,      # QUERY_ALIAS
    0x8,      # QUERY_TYPE
    0x10,     # QUERY_LEVEL
    0x20,     # QUERY_RANK
    0x40,     # QUERY_ATTRIBUTE
    0x100,    # QUERY_ATTACK
    0x200,    # QUERY_DEFENSE
    0x400,    # QUERY_BASE_ATTACK
    0x800,    # QUERY_BASE_DEFENSE
    0x1000,   # QUERY_REASON
    0x2000000, # QUERY_COVER
    0x80000,  # QUERY_STATUS
    0x200000, # QUERY_LSCALE
    0x400000, # QUERY_RSCALE
}

_FLAG_TO_KEY = {
    0x1: "code",
    0x2: "position",
    0x4: "alias",
    0x8: "type",
    0x10: "level",
    0x20: "rank",
    0x40: "attribute",
    0x80: "race",
    0x100: "attack",
    0x200: "defense",
    0x400: "base_attack",
    0x800: "base_defense",
    0x1000: "reason",
    0x40000: "owner",
    0x80000: "status",
    0x100000: "is_public",
    0x200000: "lscale",
    0x400000: "rscale",
    0x1000000: "is_hidden",
    0x2000000: "cover",
}


def _parse_query_location(data: bytes) -> list[dict]:
    """Parse an OCG_DuelQueryLocation buffer (edo9300 format).

    Returns a list of card dicts (one per slot). Empty slots are empty dicts.
    Cards include 'sequence' set to their index in the list.
    """
    if len(data) < 4:
        return []
    total_data = struct.unpack_from("<I", data, 0)[0]
    cards: list[dict] = []
    pos = 4
    end = 4 + total_data
    seq = 0

    while pos < end:
        # Read the first uint16 — if 0, empty slot
        if pos + 2 > end:
            break
        first_u16 = struct.unpack_from("<h", data, pos)[0]
        if first_u16 == 0:
            cards.append({})
            pos += 2
            seq += 1
            continue

        # Non-zero: this is the field_size of the first field block
        card: dict = {"sequence": seq}
        while pos < end:
            if pos + 2 > end:
                break
            field_size = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            if pos + field_size > end:
                break
            if field_size < 4:
                break
            flag = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            data_size = field_size - 4

            if flag == QUERY_END:
                break

            key = _FLAG_TO_KEY.get(flag)
            if key and flag in _SIMPLE_U32_FLAGS and data_size >= 4:
                card[key] = struct.unpack_from("<i" if flag in (0x100, 0x200, 0x400, 0x800) else "<I", data, pos)[0]
            elif flag == 0x80 and data_size >= 8:  # QUERY_RACE (uint64)
                card["race"] = struct.unpack_from("<Q", data, pos)[0]
            elif flag == 0x40000 and data_size >= 1:  # QUERY_OWNER (uint8)
                card["owner"] = data[pos]
            elif flag == 0x100000 and data_size >= 1:  # QUERY_IS_PUBLIC (uint8)
                card["is_public"] = data[pos]
            elif flag == 0x1000000 and data_size >= 1:  # QUERY_IS_HIDDEN (uint8)
                card["is_hidden"] = data[pos]
            elif flag == 0x800000 and data_size >= 8:  # QUERY_LINK
                card["link_rating"] = struct.unpack_from("<I", data, pos)[0]
                card["link_marker"] = struct.unpack_from("<I", data, pos + 4)[0]
            elif flag == 0x10000 and data_size >= 4:  # QUERY_OVERLAY_CARD
                count = struct.unpack_from("<I", data, pos)[0]
                overlays = []
                for j in range(count):
                    off = pos + 4 + j * 4
                    if off + 4 <= pos + data_size:
                        overlays.append(struct.unpack_from("<I", data, off)[0])
                card["overlay_cards"] = overlays
            elif flag == 0x20000 and data_size >= 4:  # QUERY_COUNTERS
                count = struct.unpack_from("<I", data, pos)[0]
                counters = []
                for j in range(count):
                    off = pos + 4 + j * 4
                    if off + 4 <= pos + data_size:
                        counters.append(struct.unpack_from("<I", data, off)[0])
                card["counters"] = counters

            pos += data_size

        cards.append(card)
        seq += 1

    return cards


def _query_location(duel: Duel, player: int, location: int) -> list[dict]:
    """Query a location using the correct edo9300 buffer parser."""
    if duel._duel_handle is None:
        return []
    info = OCG_QueryInfo()
    info.flags = QUERY_BASIC
    info.con = player
    info.loc = location
    info.seq = 0
    info.overlay_seq = 0

    length = c_uint32()
    buf_ptr = duel._lib.OCG_DuelQueryLocation(
        duel._duel_handle, ctypes.byref(length), ctypes.byref(info)
    )
    if length.value > 0 and buf_ptr:
        buf = ctypes.string_at(buf_ptr, length.value)
        return _parse_query_location(buf)
    return []


# ─── Board state builder ──────────────────────────────────────────────────


def _card_type_str(type_val: int) -> str:
    if type_val & TYPE_MONSTER:
        return "monster"
    if type_val & TYPE_SPELL:
        return "spell"
    if type_val & TYPE_TRAP:
        return "trap"
    return "unknown"


def _build_card(card: dict, card_db, hidden: bool = False) -> dict:
    """Build a single card dict. If hidden, mask code/name/stats."""
    code = card.get("code", 0)
    position = card.get("position", 0)
    pos_str = _POS_NAMES.get(position, "ATK")
    is_facedown = bool(position & POS_FACEDOWN)

    if hidden and is_facedown:
        return {
            "code": 0,
            "name": "Face-down card",
            "type": "unknown",
            "position": pos_str,
            "attack": None,
            "defense": None,
            "level": 0,
        }

    name = card_db.get_card_name(code) if code else "Unknown"
    db_card = card_db.get_card(code) if code else None
    type_val = card.get("type", 0) or (db_card["type"] if db_card else 0)

    result: dict = {
        "code": code,
        "name": name,
        "type": _card_type_str(type_val),
        "position": pos_str,
        "attack": card.get("attack"),
        "defense": card.get("defense"),
        "level": card.get("level", 0) or card.get("rank", 0),
    }
    if type_val & TYPE_LINK:
        result["link_rating"] = card.get("link_rating", 0) or (db_card["level"] if db_card else 0)
    return result


def _build_hand_card(card: dict, card_db) -> dict:
    code = card.get("code", 0)
    name = card_db.get_card_name(code) if code else "Unknown"
    db_card = card_db.get_card(code) if code else None
    type_val = card.get("type", 0) or (db_card["type"] if db_card else 0)

    result: dict = {
        "code": code,
        "name": name,
        "type": _card_type_str(type_val),
    }
    if type_val & TYPE_MONSTER:
        result["attack"] = card.get("attack") if card.get("attack") is not None else (db_card["attack"] if db_card else None)
        result["defense"] = card.get("defense") if card.get("defense") is not None else (db_card["defense"] if db_card else None)
        result["level"] = card.get("level", 0) or card.get("rank", 0) or (db_card["level"] if db_card else 0)
    return result


def _build_zone(cards: list[dict], card_db, num_slots: int, hidden: bool = False) -> list[dict | None]:
    """Build a fixed-length zone list (None = empty slot)."""
    zone: list[dict | None] = [None] * num_slots
    for card in cards:
        if not card.get("code"):
            continue
        seq = card.get("sequence", 0)
        if 0 <= seq < num_slots:
            zone[seq] = _build_card(card, card_db, hidden=hidden)
    return zone


def build_board_state(env: YuGiOhEnvironment) -> dict:
    """Build the full board state dict from a live environment."""
    duel = env._duel
    card_db = env._card_db
    agent = env._agent_player
    opp = 1 - agent

    if duel is None:
        return {"player": {}, "opponent": {}}

    # Query all zones using the correct parser
    agent_hand = _query_location(duel, agent, LOCATION_HAND)
    agent_monsters = _query_location(duel, agent, LOCATION_MZONE)
    agent_st = _query_location(duel, agent, LOCATION_SZONE)
    agent_grave = _query_location(duel, agent, LOCATION_GRAVE)
    agent_banished = _query_location(duel, agent, LOCATION_BANISHED)

    opp_monsters = _query_location(duel, opp, LOCATION_MZONE)
    opp_st = _query_location(duel, opp, LOCATION_SZONE)
    opp_grave = _query_location(duel, opp, LOCATION_GRAVE)
    opp_banished = _query_location(duel, opp, LOCATION_BANISHED)

    gs = duel.game_state

    # Build agent side (full info)
    agent_st_zone = _build_zone(agent_st, card_db, 6)
    player = {
        "hand": [_build_hand_card(c, card_db) for c in agent_hand if c.get("code")],
        "monsters": _build_zone(agent_monsters, card_db, 5),
        "spells_traps": agent_st_zone[:5],
        "field_zone": agent_st_zone[5],
        "graveyard": [_build_card(c, card_db) for c in agent_grave if c.get("code")],
        "banished": [_build_card(c, card_db) for c in agent_banished if c.get("code")],
        "extra_deck_count": gs.extra_count[agent],
        "deck_count": gs.deck_count[agent],
        "lp": gs.lp[agent],
    }

    # Build opponent side (face-down cards hidden)
    opp_st_zone = _build_zone(opp_st, card_db, 6, hidden=True)
    opponent = {
        "hand_count": gs.hand_count[opp],
        "monsters": _build_zone(opp_monsters, card_db, 5, hidden=True),
        "spells_traps": opp_st_zone[:5],
        "field_zone": opp_st_zone[5],
        "graveyard": [_build_card(c, card_db) for c in opp_grave if c.get("code")],
        "banished": [_build_card(c, card_db) for c in opp_banished if c.get("code")],
        "extra_deck_count": gs.extra_count[opp],
        "deck_count": gs.deck_count[opp],
        "lp": gs.lp[opp],
    }

    return {"player": player, "opponent": opponent}
