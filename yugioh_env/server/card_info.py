"""Decode a cards.cdb row into the printed card face for the detail widget.

Pure code→model mapping: no FastAPI, no engine, no duel state. All TYPE_* bit
decoding lives here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from yugioh_core.constants import (
    ATTRIBUTE_NAMES,
    LINK_MARKER_NAMES,
    MONSTER_TYPE_LABELS,
    RACE_NAMES,
    SPELL_TRAP_TYPE_LABELS,
    TYPE_LINK,
    TYPE_NORMAL,
    TYPE_PENDULUM,
    TYPE_TOKEN,
    TYPE_XYZ,
    CardTypeName,
    card_type_name,
)

LevelKind = Literal["level", "rank", "link"]


class CardScales(BaseModel):
    """Pendulum scales, present only on Pendulum monsters."""

    left: int
    right: int


class CardInfo(BaseModel):
    """The printed face of one card, as served by GET /api/web/card/{code}.

    Optional fields are explicit nulls rather than omitted keys.
    """

    code: int
    name: str
    desc: str
    card_type: CardTypeName
    typeline: list[str]
    attribute: str | None = None
    race: str | None = None
    level: int | None = None
    level_kind: LevelKind | None = None
    attack: int | None = None
    defense: int | None = None
    scales: CardScales | None = None
    link_arrows: list[str] | None = None


def _monster_typeline(type_val: int, race_label: str | None) -> list[str]:
    labels = [race_label] if race_label else []
    is_token = bool(type_val & TYPE_TOKEN)
    for mask, label in MONSTER_TYPE_LABELS:
        if not type_val & mask:
            continue
        # Token cards print "Machine / Token", never "… / Normal", even though
        # cards.cdb sets TYPE_NORMAL on all of them. Other labels are kept:
        # Swordsoul Token is "Wyrm / Token / Tuner".
        if is_token and mask == TYPE_NORMAL:
            continue
        labels.append(label)
    return labels


def _spell_trap_typeline(card_type: CardTypeName, type_val: int) -> list[str]:
    labels = ["Spell" if card_type == "spell" else "Trap"]
    for mask, label in SPELL_TRAP_TYPE_LABELS:
        if type_val & mask:
            labels.append(label)
    return labels


def _level_kind(type_val: int) -> LevelKind:
    if type_val & TYPE_XYZ:
        return "rank"
    if type_val & TYPE_LINK:
        return "link"
    return "level"


def _link_arrows(link_marker: int) -> list[str]:
    return [name for mask, name in LINK_MARKER_NAMES.items() if link_marker & mask]


def build_card_info(code: int, card_db) -> CardInfo | None:
    """Build the printed face for `code`, or None when it isn't in the database."""
    row = card_db.get_card(code)
    if row is None:
        return None

    type_val = row["type"]
    card_type = card_type_name(type_val)
    race_label = RACE_NAMES.get(row["race"]) if row["race"] else None

    # Stats stay None for anything that isn't a monster: spell/trap rows can
    # carry real race and attribute values, and printing them would put a bare
    # "WATER" line above "Trap".
    attribute = race = level_kind = None
    level = attack = defense = None
    scales = None
    link_arrows = None

    if card_type == "monster":
        is_link = bool(type_val & TYPE_LINK)
        typeline = _monster_typeline(type_val, race_label)
        attribute = ATTRIBUTE_NAMES.get(row["attribute"]) if row["attribute"] else None
        race = race_label
        level = row["level"]
        level_kind = _level_kind(type_val)
        attack = row["attack"]
        defense = row["defense"]  # already None for Link monsters
        if is_link:
            link_arrows = _link_arrows(row["link_marker"])
        if type_val & TYPE_PENDULUM:
            scales = CardScales(left=row["lscale"], right=row["rscale"])
    elif card_type in ("spell", "trap"):
        typeline = _spell_trap_typeline(card_type, type_val)
    else:
        # No structural bit — no table to resolve against.
        typeline = []

    return CardInfo(
        code=code,
        name=card_db.get_card_name(code),
        desc=card_db.get_card_desc(code) or "",
        card_type=card_type,
        typeline=typeline,
        attribute=attribute,
        race=race,
        level=level,
        level_kind=level_kind,
        attack=attack,
        defense=defense,
        scales=scales,
        link_arrows=link_arrows,
    )
