"""Single source of truth for human-readable action and prompt labels.

Consumed by both the web API and the CLI. Reads the wire-format
YuGiOhObservation directly — no in-process mapper dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import (
    ATTRIBUTE_NAMES,
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_OVERLAY,
    LOCATION_SZONE,
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
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    RACE_NAMES,
    RPS_NAMES,
)
from yugioh_core.string_resolver import CardTextResolver
from yugioh_env.models import (
    ActivateEffect,
    AnnounceCard,
    AnnounceNumber,
    Attack,
    CardCommand,
    ChooseOption,
    ChoosePosition,
    ChooseRPS,
    Confirm,
    FinishPick,
    Pass,
    PhaseChange,
    PickBit,
    PickCard,
    PlaceZone,
    SelectCounter,
    YuGiOhObservation,
)

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

_PROMPT_TYPE_MAP = {
    MSG_SELECT_IDLECMD: "idle_cmd",
    MSG_SELECT_BATTLECMD: "battle_cmd",
    MSG_SELECT_EFFECTYN: "effect_yn",
    MSG_SELECT_YESNO: "yes_no",
    MSG_SELECT_OPTION: "option",
    MSG_SELECT_CARD: "select_card",
    MSG_SELECT_CHAIN: "chain_link",
    MSG_SELECT_PLACE: "place",
    MSG_SELECT_DISFIELD: "place",
    MSG_SELECT_POSITION: "position",
    MSG_SELECT_TRIBUTE: "tribute",
    MSG_SELECT_UNSELECT_CARD: "select_card",
    MSG_ANNOUNCE_NUMBER: "number",
    MSG_ANNOUNCE_CARD: "announce_card",
    MSG_ANNOUNCE_RACE: "race",
    MSG_ANNOUNCE_ATTRIB: "attribute",
    MSG_ROCK_PAPER_SCISSORS: "rps",
    MSG_SELECT_COUNTER: "counter",
    MSG_SORT_CARD: "sort_card",
    MSG_SORT_CHAIN: "sort_card",
}


# Maps engine LOCATION_* bits to the human-readable strings ygopro uses to
# substitute the second `%ls` in prompt templates.
_LOCATION_NAMES = {
    LOCATION_DECK: "Deck",
    LOCATION_HAND: "Hand",
    LOCATION_MZONE: "Monster Zone",
    LOCATION_SZONE: "Spell/Trap Zone",
    LOCATION_GRAVE: "Graveyard",
    LOCATION_BANISHED: "Banished",
    LOCATION_EXTRA: "Extra Deck",
    LOCATION_OVERLAY: "Xyz Material",
}


def _format_prompt_text(template: str, card_name: str, location: int) -> str | None:
    """Substitute printf-style placeholders in a sysstring prompt template.

    strings.conf stores prompts with `%ls` for card/location names. ygopro
    clients substitute these at render time using prompt-specific context.
    For the prompts we expose (yes/no, effect-yn), the order is always
    (card_name, location_name).

    Returns None if a placeholder remains unfilled (e.g. a `%d` we don't
    know how to fill) so the caller can fall back to a synthesized question
    rather than show the raw template.
    """
    text = template
    # count=1 on each replace preserves the (card, location) slot order.
    if "%ls" in text and card_name:
        text = text.replace("%ls", card_name, 1)
    if "%ls" in text:
        loc_name = _LOCATION_NAMES.get(location, "")
        if loc_name:
            text = text.replace("%ls", loc_name, 1)
    if "%ls" in text or "%d" in text or "%s" in text:
        return None
    return text


@dataclass
class ActionDetails:
    """Per-action descriptor consumed by both the web API and the CLI.

    Mirrors the dict shape the previous server-side describe_actions
    emitted, with `controller` flowing through unchanged (already
    relativized at extraction: 0=mine, 1=opp).
    """

    index: int
    description: str
    category: str
    card_code: int
    card_name: str
    controller: int
    location: int
    sequence: int

    def to_dict(self) -> dict:
        return asdict(self)


class ActionDescriber:
    """Single-source describer for action labels and prompt metadata."""

    def __init__(
        self,
        card_db: CardDatabase,
        sys_strings: dict[int, str] | None = None,
    ) -> None:
        self._text = CardTextResolver(card_db, sys_strings=sys_strings)

    def describe(self, obs: YuGiOhObservation, action_idx: int) -> ActionDetails:
        """Describe a single legal action by slot index. Raises IndexError
        for out-of-range or inactive slots."""
        if action_idx < 0 or action_idx >= len(obs.action_mask):
            raise IndexError(f"action_idx {action_idx} out of range (len={len(obs.action_mask)})")
        if obs.action_mask[action_idx] != 1:
            raise IndexError(f"action slot {action_idx} is inactive (action_mask=0)")
        return self._describe_one(obs, action_idx)

    def describe_all(self, obs: YuGiOhObservation) -> list[ActionDetails]:
        """One ActionDetails per legal action (where action_mask[i] == 1)."""
        return [self._describe_one(obs, i) for i, m in enumerate(obs.action_mask) if m == 1]

    def describe_prompt(self, obs: YuGiOhObservation) -> dict | None:
        """Build the prompt-level metadata dict for the current observation,
        or None when there's no active prompt."""
        if obs.prompt_meta is None:
            return None
        meta = dict(obs.prompt_meta)
        # msg_type is carried in prompt_meta (set by the env's
        # _build_prompt_meta from mapper.msg_type); it is a documented,
        # always-present field of the prompt_meta wire contract.
        msg_type = meta.pop("msg_type", None)
        prompt_type = _PROMPT_TYPE_MAP.get(msg_type, "unknown")
        result: dict = {"type": prompt_type, **meta}
        # card_name lookup happens here (the server-side _build_prompt_meta
        # only sets card_code; we expand it to include card_name for
        # prompts that carry one).
        if "card_code" in result:
            code = result["card_code"]
            result["card_name"] = self._text.card_name(code)
        # Resolve the prompt-level desc (yes/no, effect-yn) to display text.
        # desc == 0 means "no specific prompt string"; an unknown id means the
        # resolver can't find it; a template with placeholders we can't fill
        # also yields None. In all those cases the consumer falls back to a
        # synthesized question.
        if "desc" in result:
            desc = result["desc"]
            template = self._text.effect_text(desc)
            if template is None:
                result["prompt_text"] = None
            else:
                result["prompt_text"] = _format_prompt_text(
                    template,
                    card_name=result.get("card_name", ""),
                    location=result.get("location", 0),
                )
        return result

    # ─── Internal dispatch ────────────────────────────────────────────────

    def _describe_one(self, obs: YuGiOhObservation, idx: int) -> ActionDetails:
        descriptors = obs.action_descriptors
        d = descriptors[idx] if idx < len(descriptors) else None
        if d is None:
            raise IndexError(f"action slot {idx} has no descriptor")
        msg_type = obs.msg_type

        card_code = 0
        controller = 0
        location = 0
        sequence = 0

        # Only 5 of 16 variants carry a CardRef (PickCard, CardCommand,
        # ActivateEffect, Attack, SelectCounter); pull coordinates from it
        # when present.
        card_ref = getattr(d, "card", None)
        if card_ref is not None:
            card_code = card_ref.code
            controller = card_ref.controller
            location = card_ref.location
            sequence = card_ref.sequence
        elif isinstance(d, PlaceZone):
            controller = d.controller
            location = d.location
            sequence = d.sequence
        elif isinstance(d, Confirm) and msg_type == MSG_SELECT_EFFECTYN:
            # EFFECTYN is a re-sourcing, not a value change: Confirm carries
            # no card fields of its own, so the four card fields come from
            # prompt_meta (already relativized by _build_prompt_meta).
            meta = obs.prompt_meta or {}
            card_code = meta.get("card_code", 0)
            controller = meta.get("controller", 0)
            location = meta.get("location", 0)
            sequence = meta.get("sequence", 0)
        elif isinstance(d, AnnounceCard | ChoosePosition):
            card_code = d.card_code

        card_name = self._text.card_name(card_code)

        description, category = self._dispatch(msg_type, d, idx, card_name)

        return ActionDetails(
            index=idx,
            description=description,
            category=category,
            card_code=card_code,
            card_name=card_name,
            controller=controller,
            location=location,
            sequence=sequence,
        )

    def _dispatch(self, msg_type: int, d, idx: int, card_name: str) -> tuple[str, str]:
        """Describe one action. Keyed on (msg_type, variant): variants are
        deliberately shared across prompts, so the variant supplies the
        *fields* and msg_type supplies the *wording*.

        NOTE: bare uppercase names (e.g. ``MSG_SELECT_IDLECMD``) are NOT
        value patterns in Python's `match` grammar -- an undotted identifier
        in a case pattern is always a *capture* pattern (binds unconditionally,
        matching any value) unless it is a dotted/attribute name or a literal.
        Writing `case (MSG_SELECT_IDLECMD, CardCommand()):` would therefore
        match on the variant type alone and ignore msg_type entirely --
        exactly the shared-variant collapse this dispatch exists to avoid.
        Every msg_type comparison below is done via an explicit `if` guard.
        """
        match (msg_type, d):
            case (mt, CardCommand()) if mt == MSG_SELECT_IDLECMD:
                label, cat_str = _IDLE_DESCS.get(d.command, (f"Action {d.command}", "unknown"))
                return f"{label} {card_name}".rstrip(), cat_str

            case (mt, ActivateEffect()) if mt == MSG_SELECT_IDLECMD:
                label, cat_str = _IDLE_DESCS[IDLE_ACTIVATE]
                base = f"{label} {card_name}".rstrip()
                resolved = self._text.effect_text(d.desc)
                return (f"{base}: {resolved}" if resolved else base), cat_str

            case (mt, PhaseChange()) if mt == MSG_SELECT_IDLECMD:
                cat = IDLE_TO_BP if d.to == "bp" else IDLE_TO_EP
                label, cat_str = _IDLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
                return label, cat_str

            case (mt, ActivateEffect()) if mt == MSG_SELECT_BATTLECMD:
                label, cat_str = _BATTLE_DESCS[BATTLE_ACTIVATE]
                base = f"{label} {card_name}".rstrip()
                resolved = self._text.effect_text(d.desc)
                return (f"{base}: {resolved}" if resolved else base), cat_str

            case (mt, Attack()) if mt == MSG_SELECT_BATTLECMD:
                label, cat_str = _BATTLE_DESCS[BATTLE_ATTACK]
                return f"{label} {card_name}".rstrip(), cat_str

            case (mt, PhaseChange()) if mt == MSG_SELECT_BATTLECMD:
                cat = BATTLE_TO_M2 if d.to == "m2" else BATTLE_TO_EP
                label, cat_str = _BATTLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
                return label, cat_str

            case (mt, Confirm()) if mt in (MSG_SELECT_EFFECTYN, MSG_SELECT_YESNO):
                return ("Yes", "yes") if d.yes else ("No", "no")

            case (mt, ChooseOption()) if mt == MSG_SELECT_OPTION:
                resolved = self._text.effect_text(d.desc)
                return (resolved or f"effect 0x{d.desc:x}"), "option"

            case (mt, FinishPick()) if mt == MSG_SELECT_CARD:
                n = d.num_selected
                return f"Finish selecting ({n} card{'s' if n != 1 else ''})", "finish"

            case (mt, PickCard()) if mt in (MSG_SELECT_CARD, MSG_SELECT_UNSELECT_CARD):
                label = f"Select {card_name}" if card_name else f"Select card #{d.engine_index}"
                return label, "select_card"

            case (mt, Pass()) if mt == MSG_SELECT_CHAIN:
                return "Pass (no chain)", "pass"

            case (mt, ActivateEffect()) if mt == MSG_SELECT_CHAIN:
                target = card_name or f"#{d.engine_index}"
                resolved = self._text.effect_text(d.desc)
                if resolved and card_name:
                    return f"Chain {target}: {resolved}", "chain"
                return f"Chain {target}", "chain"

            case (mt, PlaceZone()) if mt in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
                zone = (
                    "Monster"
                    if d.location == LOCATION_MZONE
                    else "Spell/Trap"
                    if d.location == LOCATION_SZONE
                    else "Unknown"
                )
                return f"Place in {zone} Zone {d.sequence + 1}", "place"

            case (mt, ChoosePosition()) if mt == MSG_SELECT_POSITION:
                pos_name = _POS_NAMES.get(d.position, f"Position {d.position}")
                desc = f"{card_name}: {pos_name}" if card_name else pos_name
                return desc, "position"

            case (mt, FinishPick()) if mt == MSG_SELECT_TRIBUTE:
                n = d.num_selected
                return f"Finish tributing ({n} card{'s' if n != 1 else ''})", "finish"

            case (mt, PickCard()) if mt == MSG_SELECT_TRIBUTE:
                label = f"Tribute {card_name}" if card_name else f"Tribute card #{d.engine_index}"
                return label, "tribute"

            case (mt, Pass()) if mt == MSG_SELECT_UNSELECT_CARD:
                return "Finish selection", "finish"

            case (mt, AnnounceCard()) if mt == MSG_ANNOUNCE_CARD:
                label = f"Declare {card_name}" if card_name else f"Declare card #{idx}"
                return label, "announce_card"

            case (mt, AnnounceNumber()) if mt == MSG_ANNOUNCE_NUMBER:
                return f"Announce {d.value}", "number"

            case (mt, PickBit()) if mt == MSG_ANNOUNCE_RACE:
                return RACE_NAMES.get(d.value, f"Race(0x{d.value:x})"), "race"

            case (mt, PickBit()) if mt == MSG_ANNOUNCE_ATTRIB:
                return ATTRIBUTE_NAMES.get(d.value, f"Attr(0x{d.value:x})"), "attribute"

            case (mt, ChooseRPS()) if mt == MSG_ROCK_PAPER_SCISSORS:
                return RPS_NAMES[d.choice], "rps"

            case (mt, SelectCounter()) if mt == MSG_SELECT_COUNTER:
                target = card_name or f"card #{d.engine_index}"
                return f"Remove {d.counter_count} from {target}", "counter"

            case (mt, PickCard()) if mt in (MSG_SORT_CARD, MSG_SORT_CHAIN):
                label = (
                    f"Place {card_name} next" if card_name else f"Place card #{d.engine_index} next"
                )
                return label, "sort"

            case (mt, PickCard()) if mt == MSG_SELECT_SUM:
                # No dedicated wording was ever written for MSG_SELECT_SUM;
                # it fell through to the generic fallback below. Preserved
                # verbatim rather than "fixed" here (out of scope).
                if card_name:
                    return f"Select {card_name}", "select_card"
                return f"Action #{d.engine_index}", "unknown"

            case (mt, FinishPick()) if mt == MSG_SELECT_SUM:
                # The harness's finish action always carries index=0.
                return "Action #0", "unknown"

            case _:
                if card_name:
                    return f"Select {card_name}", "select_card"
                return f"Action #{idx}", "unknown"
