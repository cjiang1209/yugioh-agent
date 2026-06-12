"""Puzzle state validation, Lua generation, and file loading."""

from __future__ import annotations

import json
from pathlib import Path

from yugioh_core.constants import (
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)

_POSITION_MAP: dict[str, int] = {
    "face_up_attack": POS_FACEUP_ATTACK,
    "face_down_attack": POS_FACEDOWN_ATTACK,
    "face_up_defense": POS_FACEUP_DEFENSE,
    "face_down_defense": POS_FACEDOWN_DEFENSE,
    "face_up": POS_FACEUP,
    "face_down": POS_FACEDOWN,
}

_VALID_PLAYER_KEYS = {
    "lp",
    "hand",
    "monster_zone",
    "spell_zone",
    "grave",
    "banished",
    "deck",
    "extra",
}

_FIELD_ZONES: dict[str, int] = {
    "monster_zone": 6,  # seq 0-4 main, 5-6 extra monster zones
    "spell_zone": 5,  # seq 0-4 spell/trap, 5 field spell
}

_LIST_ZONES = ("hand", "grave", "banished", "deck", "extra")

_DISABLE_TEMPLATE = """\
do
  local e=Effect.GlobalEffect()
  e:SetType(EFFECT_TYPE_FIELD)
  e:SetCode(EFFECT_DISABLE)
  e:SetTargetRange(LOCATION_MZONE+LOCATION_SZONE,LOCATION_MZONE+LOCATION_SZONE)
  e:SetTarget(function(e,c) return c:GetControler()=={con} and c:GetSequence()=={seq} and c:IsLocation({loc}) end)
  Duel.RegisterEffect(e,0)
end"""


def parse_position(pos: str) -> int:
    """Map a position string to the corresponding engine constant.

    Raises ``ValueError`` on unrecognised strings.
    """
    try:
        return _POSITION_MAP[pos]
    except KeyError:
        raise ValueError(f"Unknown position '{pos}'; valid: {sorted(_POSITION_MAP)}") from None


def _default_player() -> dict:
    return {
        "lp": 8000,
        "hand": [],
        "monster_zone": [],
        "spell_zone": [],
        "grave": [],
        "banished": [],
        "deck": [],
        "extra": [],
    }


def _validate_card_code(code: object, context: str) -> int:
    """Ensure *code* is a positive integer card code."""
    if not isinstance(code, int) or code <= 0:
        raise ValueError(f"Card code in '{context}' must be a positive int, got {code!r}")
    return code


def _validate_card_list(cards: list, zone_name: str) -> list[int]:
    return [_validate_card_code(c, zone_name) for c in cards]


def _validate_field_zone(
    cards: list[dict],
    zone_name: str,
    max_seq: int,
) -> list[dict]:
    validated: list[dict] = []
    seen_seqs: set[int] = set()

    for card in cards:
        code = _validate_card_code(card["code"], zone_name)
        pos = parse_position(card["pos"])
        seq = card["seq"]
        if not isinstance(seq, int) or seq < 0 or seq > max_seq:
            raise ValueError(f"'{zone_name}' seq must be 0-{max_seq}, got {seq}")
        if seq in seen_seqs:
            raise ValueError(f"Duplicate seq {seq} in '{zone_name}'")
        seen_seqs.add(seq)
        disabled = bool(card.get("disabled", False))
        validated.append({"code": code, "pos": pos, "seq": seq, "disabled": disabled})

    return validated


def _validate_player(player: dict, label: str) -> dict:
    unknown = set(player) - _VALID_PLAYER_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {sorted(unknown)}")
    defaults = _default_player()
    lp = player.get("lp", defaults["lp"])
    if not isinstance(lp, int) or lp < 0:
        raise ValueError(f"LP must be a non-negative int, got {lp!r}")
    defaults["lp"] = lp

    for zone in _LIST_ZONES:
        if zone in player:
            defaults[zone] = _validate_card_list(player[zone], zone)

    for zone, max_seq in _FIELD_ZONES.items():
        if zone in player:
            defaults[zone] = _validate_field_zone(player[zone], zone, max_seq=max_seq)

    return defaults


def validate_puzzle(state: dict) -> dict:
    """Validate and normalise a puzzle state dict.

    Missing players default to an empty board with 8000 LP.
    Position strings in field zones are converted to engine constants.

    Raises ``ValueError`` on invalid data.
    """
    p0 = _validate_player(state.get("player0", {}), "player0")
    p1 = _validate_player(state.get("player1", {}), "player1")

    return {"player0": p0, "player1": p1}


def generate_disable_lua(state: dict) -> str | None:
    """Generate Lua source to disable marked cards.

    Returns ``None`` if no cards have ``disabled: True``.
    """
    blocks: list[str] = []

    _ZONE_TO_LOC = {"monster_zone": "LOCATION_MZONE", "spell_zone": "LOCATION_SZONE"}

    for player_idx in (0, 1):
        player = state[f"player{player_idx}"]
        for zone, loc in _ZONE_TO_LOC.items():
            for card in player.get(zone, []):
                if card.get("disabled"):
                    blocks.append(
                        _DISABLE_TEMPLATE.format(
                            con=player_idx,
                            seq=card["seq"],
                            loc=loc,
                        )
                    )

    if not blocks:
        return None
    return "\n".join(blocks) + "\n"


def load_puzzle(path: str | Path) -> dict:
    """Load a puzzle from a JSON file.

    Raises ``ValueError`` on unsupported file extensions.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported puzzle file extension: {suffix}")

    return validate_puzzle(raw)
