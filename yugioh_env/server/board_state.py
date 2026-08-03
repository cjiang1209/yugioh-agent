"""Render a captured raw board dict into the human-readable display board."""

from __future__ import annotations

from yugioh_core.constants import (
    POS_ATTACK,
    POS_DEFENSE,
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    TYPE_LINK,
    TYPE_MONSTER,
    card_type_name,
)

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

# ─── Board rendering ──────────────────────────────────────────────────────


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
        "type": card_type_name(type_val),
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
        "type": card_type_name(type_val),
    }
    if type_val & TYPE_MONSTER:
        result["attack"] = (
            card.get("attack")
            if card.get("attack") is not None
            else (db_card["attack"] if db_card else None)
        )
        result["defense"] = (
            card.get("defense")
            if card.get("defense") is not None
            else (db_card["defense"] if db_card else None)
        )
        result["level"] = (
            card.get("level", 0) or card.get("rank", 0) or (db_card["level"] if db_card else 0)
        )
        if type_val & TYPE_LINK:
            result["link_rating"] = card.get("link_rating", 0) or (
                db_card["level"] if db_card else 0
            )
    return result


def _build_zone(
    cards: list[dict], card_db, num_slots: int, hidden: bool = False
) -> list[dict | None]:
    """Build a fixed-length zone list (None = empty slot)."""
    zone: list[dict | None] = [None] * num_slots
    for card in cards:
        if not card.get("code"):
            continue
        seq = card.get("sequence", 0)
        if 0 <= seq < num_slots:
            zone[seq] = _build_card(card, card_db, hidden=hidden)
    return zone


def _render_agent(side: dict, card_db) -> dict:
    st = _build_zone(side["spells_traps"], card_db, 6)
    mon = _build_zone(side["monsters"], card_db, 7)
    return {
        "hand": [_build_card_info(c, card_db) for c in side["hand"] if c.get("code")],
        "monsters": mon[:5],
        "spells_traps": st[:5],
        "field_zone": st[5],
        "extra_monster_zone": [mon[5], mon[6]],
        "graveyard": [_build_card(c, card_db) for c in side["grave"] if c.get("code")],
        "banished": [_build_card(c, card_db) for c in side["banished"] if c.get("code")],
        "extra_deck": [_build_card_info(c, card_db) for c in side["extra"] if c.get("code")],
        "deck_count": side["deck_count"],
        "lp": side["lp"],
    }


def _render_opponent(side: dict, card_db, *, open_cards: bool) -> dict:
    hidden = not open_cards
    st = _build_zone(side["spells_traps"], card_db, 6, hidden=hidden)
    mon = _build_zone(side["monsters"], card_db, 7, hidden=hidden)
    opp: dict = {
        "hand_count": side["hand_count"],
        "monsters": mon[:5],
        "spells_traps": st[:5],
        "field_zone": st[5],
        "extra_monster_zone": [mon[5], mon[6]],
        "graveyard": [_build_card(c, card_db) for c in side["grave"] if c.get("code")],
        "banished": [_build_card(c, card_db) for c in side["banished"] if c.get("code")],
        "extra_deck_count": side["extra_count"],
        "deck_count": side["deck_count"],
        "lp": side["lp"],
    }
    if open_cards:
        opp["hand"] = [_build_card_info(c, card_db) for c in side["hand"] if c.get("code")]
        opp["extra_deck"] = [_build_card_info(c, card_db) for c in side["extra"] if c.get("code")]
    return opp


def render_board(raw_board: dict, card_db, *, open_cards: bool = False) -> dict:
    """Render a captured RawBoard into the display board dict (sole render impl)."""
    if not raw_board.get("agent") and not raw_board.get("opponent"):
        return {"player": {}, "opponent": {}}
    return {
        "player": _render_agent(raw_board["agent"], card_db),
        "opponent": _render_opponent(raw_board["opponent"], card_db, open_cards=open_cards),
    }
