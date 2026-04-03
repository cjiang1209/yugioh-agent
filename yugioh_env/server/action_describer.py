"""Produce human-readable descriptions for the current action set."""

from __future__ import annotations

from yugioh_core.constants import (
    MSG_SELECT_IDLECMD,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,
    MSG_SELECT_OPTION,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_PLACE,
    MSG_SELECT_DISFIELD,
    MSG_SELECT_POSITION,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    POS_FACEUP_ATTACK,
    POS_FACEDOWN_ATTACK,
    POS_FACEUP_DEFENSE,
    POS_FACEDOWN_DEFENSE,
    LOCATION_MZONE,
    LOCATION_SZONE,
)
from yugioh_core.action_categories import (
    IDLE_SUMMON, IDLE_SP_SUMMON, IDLE_REPOSITION, IDLE_MSET,
    IDLE_SSET, IDLE_ACTIVATE, IDLE_TO_BP, IDLE_TO_EP,
    BATTLE_ACTIVATE, BATTLE_ATTACK, BATTLE_TO_M2, BATTLE_TO_EP,
)
from yugioh_core.card_database import CardDatabase
from yugioh_env.action_space import ActionMapper

_IDLE_DESCS = {
    IDLE_SUMMON: ("Normal Summon", "summon"),
    IDLE_SP_SUMMON: ("Special Summon", "special_summon"),
    IDLE_REPOSITION: ("Reposition", "reposition"),
    IDLE_MSET: ("Set Monster", "monster_set"),
    IDLE_SSET: ("Set Spell/Trap", "spell_set"),
    IDLE_ACTIVATE: ("Activate", "activate"),
    IDLE_TO_BP: ("To Battle Phase", "to_battle"),
    IDLE_TO_EP: ("To End Phase", "to_end"),
}

_BATTLE_DESCS = {
    BATTLE_ACTIVATE: ("Activate", "activate"),
    BATTLE_ATTACK: ("Attack", "attack"),
    BATTLE_TO_M2: ("To Main Phase 2", "to_main2"),
    BATTLE_TO_EP: ("To End Phase", "to_end"),
}

_POS_NAMES = {
    POS_FACEUP_ATTACK: "Face-up Attack",
    POS_FACEDOWN_ATTACK: "Face-down Attack",
    POS_FACEUP_DEFENSE: "Face-up Defense",
    POS_FACEDOWN_DEFENSE: "Face-down Defense",
}


def describe_actions(mapper: ActionMapper, card_db: CardDatabase) -> list[dict]:
    """Return a list of action dicts with human-readable descriptions."""
    actions = mapper.actions
    msg_type = mapper.msg_type
    result: list[dict] = []

    for i, action in enumerate(actions):
        code = action.get("code", 0)
        cat = action.get("category", 0)
        card_name = card_db.get_card_name(code) if code else ""

        desc, category_str = _describe_one(action, msg_type, card_name)

        result.append({
            "index": i,
            "description": desc,
            "card_code": code,
            "card_name": card_name,
            "category": category_str,
        })

    return result


def _describe_one(action: dict, msg_type: int, card_name: str) -> tuple[str, str]:
    """Return (description, category_string) for a single action."""
    cat = action.get("category", 0)
    code = action.get("code", 0)

    if msg_type == MSG_SELECT_IDLECMD:
        label, cat_str = _IDLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
        if code and card_name:
            return f"{label} {card_name}", cat_str
        return label, cat_str

    if msg_type == MSG_SELECT_BATTLECMD:
        label, cat_str = _BATTLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
        if code and card_name:
            return f"{label} {card_name}", cat_str
        return label, cat_str

    if msg_type == MSG_SELECT_EFFECTYN:
        if cat == 0:
            desc = f"Yes — activate {card_name}" if card_name else "Yes"
            return desc, "yes"
        return "No", "no"

    if msg_type == MSG_SELECT_YESNO:
        return ("Yes", "yes") if cat == 0 else ("No", "no")

    if msg_type == MSG_SELECT_OPTION:
        idx = action.get("index", 0)
        return f"Option {idx + 1}", "option"

    if msg_type == MSG_SELECT_CARD:
        if cat == 1:
            num = action.get("num_selected", 0)
            return f"Finish selecting ({num} card{'s' if num != 1 else ''})", "finish"
        label = f"Select {card_name}" if card_name else f"Select card #{action.get('index', 0)}"
        return label, "select_card"

    if msg_type == MSG_SELECT_CHAIN:
        if cat == 1:
            return "Pass (no chain)", "pass"
        desc = f"Chain {card_name}" if card_name else f"Chain #{action.get('index', 0)}"
        return desc, "chain"

    if msg_type in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        loc = action.get("location", 0)
        seq = action.get("sequence", 0)
        zone = "Monster" if loc == LOCATION_MZONE else "Spell/Trap" if loc == LOCATION_SZONE else "Unknown"
        return f"Place in {zone} Zone {seq + 1}", "place"

    if msg_type == MSG_SELECT_POSITION:
        idx = action.get("index", 0)
        pos_name = _POS_NAMES.get(idx, f"Position {idx}")
        desc = f"{pos_name}"
        if card_name:
            desc = f"{card_name}: {pos_name}"
        return desc, "position"

    if msg_type == MSG_SELECT_TRIBUTE:
        if cat == 1:
            num = action.get("num_selected", 0)
            return f"Finish tributing ({num} card{'s' if num != 1 else ''})", "finish"
        label = f"Tribute {card_name}" if card_name else f"Tribute card #{action.get('index', 0)}"
        return label, "tribute"

    if msg_type == MSG_SELECT_UNSELECT_CARD:
        if cat == 1:
            return "Finish selection", "finish"
        label = f"Select {card_name}" if card_name else f"Select card #{action.get('index', 0)}"
        return label, "select_card"

    # Fallback for sum, sort, announce, counter, rps, etc.
    if card_name:
        return f"Select {card_name}", "select_card"
    return f"Action #{action.get('index', 0)}", "unknown"


# ─── Prompt metadata ─────────────────────────────────────────────────────────

_PROMPT_TYPE_MAP = {
    MSG_SELECT_IDLECMD: "idle_cmd",
    MSG_SELECT_BATTLECMD: "battle_cmd",
    MSG_SELECT_EFFECTYN: "effect_yn",
    MSG_SELECT_YESNO: "yes_no",
    MSG_SELECT_OPTION: "option",
    MSG_SELECT_CARD: "select_card",
    MSG_SELECT_CHAIN: "chain",
    MSG_SELECT_PLACE: "place",
    MSG_SELECT_DISFIELD: "place",
    MSG_SELECT_POSITION: "position",
    MSG_SELECT_TRIBUTE: "tribute",
    MSG_SELECT_UNSELECT_CARD: "select_card",
}


def describe_prompt(mapper: ActionMapper, card_db: CardDatabase) -> dict:
    """Build prompt metadata from the current mapper state."""
    msg_type = mapper.msg_type
    msg = mapper.msg
    prompt_type = _PROMPT_TYPE_MAP.get(msg_type, "unknown")
    result: dict = {"type": prompt_type}

    if msg_type == MSG_SELECT_EFFECTYN:
        code = msg.get("code", 0)
        result["card_code"] = code
        result["card_name"] = card_db.get_card_name(code) if code else ""

    elif msg_type == MSG_SELECT_CARD:
        result["min"] = msg.get("min", 1)
        result["max"] = msg.get("max", 1)
        result["cancelable"] = bool(msg.get("cancelable", 0))
        result["selected_count"] = len(msg.get("_selected", []))

    elif msg_type == MSG_SELECT_TRIBUTE:
        # min = minimum total release value, NOT minimum card count.
        # A single monster with release_param=2 can satisfy min=2.
        # max = maximum number of cards that can be selected.
        selected = msg.get("_selected", [])
        cards = msg.get("cards", [])
        result["min_release"] = msg.get("min", 1)
        result["max_cards"] = msg.get("max", 1)
        result["cancelable"] = bool(msg.get("cancelable", 0))
        result["release_total"] = sum(
            cards[i].get("release_param", 1) for i in selected if i < len(cards)
        )
        result["cards_selected"] = len(selected)

    elif msg_type == MSG_SELECT_UNSELECT_CARD:
        result["min"] = msg.get("min", 1)
        result["max"] = msg.get("max", 1)
        result["finishable"] = bool(msg.get("finishable", 0))

    elif msg_type == MSG_SELECT_CHAIN:
        result["forced"] = bool(msg.get("forced", 0))

    elif msg_type == MSG_SELECT_POSITION:
        code = msg.get("code", 0)
        result["card_code"] = code
        result["card_name"] = card_db.get_card_name(code) if code else ""

    elif msg_type in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        result["count"] = msg.get("count", 1)

    return result
