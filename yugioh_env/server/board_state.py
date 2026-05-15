"""Build a human-readable board state dict from the live duel."""

from __future__ import annotations

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
    TYPE_MONSTER,
    TYPE_SPELL,
    TYPE_TRAP,
    TYPE_LINK,
)

if TYPE_CHECKING:
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


def _build_card_info(card: dict, card_db) -> dict:
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
        if type_val & TYPE_LINK:
            result["link_rating"] = card.get("link_rating", 0) or (db_card["level"] if db_card else 0)
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


def build_board_state(env: YuGiOhEnvironment, *, open_cards: bool = False) -> dict:
    """Build the full board state dict from a live environment.

    Args:
        open_cards: When True, the ``opponent`` dict includes full card data
            (unhidden face-down cards, hand contents, extra deck contents)
            for UI display.  Game logic never consumes this dict.
    """
    duel = env._duel
    card_db = env._card_db
    agent = env._agent_player
    opp = 1 - agent

    if duel is None:
        return {"player": {}, "opponent": {}}

    agent_hand = duel.query_location(agent, LOCATION_HAND)
    agent_monsters = duel.query_location(agent, LOCATION_MZONE)
    agent_st = duel.query_location(agent, LOCATION_SZONE)
    agent_grave = duel.query_location(agent, LOCATION_GRAVE)
    agent_banished = duel.query_location(agent, LOCATION_BANISHED)
    agent_extra = duel.query_location(agent, LOCATION_EXTRA)

    opp_monsters = duel.query_location(opp, LOCATION_MZONE)
    opp_st = duel.query_location(opp, LOCATION_SZONE)
    opp_grave = duel.query_location(opp, LOCATION_GRAVE)
    opp_banished = duel.query_location(opp, LOCATION_BANISHED)

    gs = duel.game_state

    # Build agent side (full info)
    agent_st_zone = _build_zone(agent_st, card_db, 6)
    agent_monsters_full = _build_zone(agent_monsters, card_db, 7)
    player = {
        "hand": [_build_card_info(c, card_db) for c in agent_hand if c.get("code")],
        "monsters": agent_monsters_full[:5],
        "spells_traps": agent_st_zone[:5],
        "field_zone": agent_st_zone[5],
        "extra_monster_zone": [agent_monsters_full[5], agent_monsters_full[6]],
        "graveyard": [_build_card(c, card_db) for c in agent_grave if c.get("code")],
        "banished": [_build_card(c, card_db) for c in agent_banished if c.get("code")],
        "extra_deck": [_build_card_info(c, card_db) for c in agent_extra if c.get("code")],
        "deck_count": gs.deck_count[agent],
        "lp": gs.lp[agent],
    }

    # Build opponent side
    opp_st_zone = _build_zone(opp_st, card_db, 6, hidden=not open_cards)
    opp_monsters_full = _build_zone(opp_monsters, card_db, 7, hidden=not open_cards)
    opponent: dict = {
        "hand_count": gs.hand_count[opp],
        "monsters": opp_monsters_full[:5],
        "spells_traps": opp_st_zone[:5],
        "field_zone": opp_st_zone[5],
        "extra_monster_zone": [opp_monsters_full[5], opp_monsters_full[6]],
        "graveyard": [_build_card(c, card_db) for c in opp_grave if c.get("code")],
        "banished": [_build_card(c, card_db) for c in opp_banished if c.get("code")],
        "extra_deck_count": gs.extra_count[opp],
        "deck_count": gs.deck_count[opp],
        "lp": gs.lp[opp],
    }

    if open_cards:
        opp_hand = duel.query_location(opp, LOCATION_HAND)
        opp_extra = duel.query_location(opp, LOCATION_EXTRA)
        opponent["hand"] = [_build_card_info(c, card_db) for c in opp_hand if c.get("code")]
        opponent["extra_deck"] = [_build_card_info(c, card_db) for c in opp_extra if c.get("code")]

    return {"player": player, "opponent": opponent}
