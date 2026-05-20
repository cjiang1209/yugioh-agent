"""Single source of truth for human-readable action and prompt labels.

Consumed by both the web API and the CLI. Reads the wire-format
YuGiOhObservation directly — no in-process mapper dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from yugioh_core.action_categories import (
    BATTLE_ACTIVATE, BATTLE_ATTACK, BATTLE_TO_EP, BATTLE_TO_M2,
    IDLE_ACTIVATE, IDLE_MSET, IDLE_REPOSITION, IDLE_SP_SUMMON,
    IDLE_SSET, IDLE_SUMMON, IDLE_TO_BP, IDLE_TO_EP,
)
from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import (
    LOCATION_BANISHED, LOCATION_DECK, LOCATION_EXTRA, LOCATION_GRAVE,
    LOCATION_HAND, LOCATION_MZONE, LOCATION_OVERLAY, LOCATION_SZONE,
    MSG_ANNOUNCE_ATTRIB, MSG_ANNOUNCE_NUMBER, MSG_ANNOUNCE_RACE,
    MSG_ROCK_PAPER_SCISSORS,
    MSG_SELECT_BATTLECMD, MSG_SELECT_CARD, MSG_SELECT_CHAIN,
    MSG_SELECT_COUNTER, MSG_SELECT_DISFIELD, MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD, MSG_SELECT_OPTION, MSG_SELECT_PLACE,
    MSG_SELECT_POSITION, MSG_SELECT_TRIBUTE, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    POS_FACEDOWN_ATTACK, POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE,
)
from yugioh_core.encoding import decode_u16, decode_u32
from yugioh_core.string_resolver import StringResolver
from yugioh_env.models import ActionMeta, YuGiOhObservation


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
    MSG_ANNOUNCE_RACE: "race",
    MSG_ANNOUNCE_ATTRIB: "attribute",
    MSG_ROCK_PAPER_SCISSORS: "rps",
    MSG_SELECT_COUNTER: "counter",
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
    meta: ActionMeta | None

    def to_dict(self) -> dict:
        """JSON-ready dict; `meta` is dumped via Pydantic to avoid the
        deepcopy that `dataclasses.asdict` would otherwise apply to the
        embedded BaseModel."""
        return {
            "index": self.index,
            "description": self.description,
            "category": self.category,
            "card_code": self.card_code,
            "card_name": self.card_name,
            "controller": self.controller,
            "location": self.location,
            "sequence": self.sequence,
            "meta": self.meta.model_dump() if self.meta is not None else None,
        }


class ActionDescriber:
    """Single-source describer for action labels and prompt metadata."""

    def __init__(
        self,
        card_db: CardDatabase,
        sys_strings: dict[int, str] | None = None,
    ) -> None:
        self._card_db = card_db
        self._resolver: StringResolver | None = (
            StringResolver(card_db, sys_strings=sys_strings)
            if sys_strings is not None else None
        )

    def describe(self, obs: YuGiOhObservation, action_idx: int) -> ActionDetails:
        """Describe a single legal action by slot index. Raises IndexError
        for out-of-range or inactive slots."""
        if action_idx < 0 or action_idx >= len(obs.action_mask):
            raise IndexError(
                f"action_idx {action_idx} out of range (len={len(obs.action_mask)})"
            )
        if obs.action_mask[action_idx] != 1:
            raise IndexError(
                f"action slot {action_idx} is inactive (action_mask=0)"
            )
        return self._describe_one(obs, action_idx)

    def describe_all(self, obs: YuGiOhObservation) -> list[ActionDetails]:
        """One ActionDetails per legal action (where action_mask[i] == 1)."""
        return [
            self._describe_one(obs, i)
            for i, m in enumerate(obs.action_mask)
            if m == 1
        ]

    def describe_prompt(self, obs: YuGiOhObservation) -> dict | None:
        """Build the prompt-level metadata dict for the current observation,
        or None when there's no active prompt."""
        if obs.prompt_meta is None:
            return None
        meta = dict(obs.prompt_meta)
        # Prefer msg_type carried in prompt_meta (set by the env's
        # _build_prompt_meta from mapper.msg_type) — falls back to byte 0
        # of the first action vector when the field is absent (legacy callers
        # that hand-built obs).
        msg_type = meta.pop("msg_type", None)
        if msg_type is None:
            if not obs.actions:
                return None
            msg_type = obs.actions[0][0]
        prompt_type = _PROMPT_TYPE_MAP.get(msg_type, "unknown")
        result: dict = {"type": prompt_type, **meta}
        # card_name lookup happens here (the server-side _build_prompt_meta
        # only sets card_code; we expand it to include card_name for
        # prompts that carry one).
        if "card_code" in result:
            code = result["card_code"]
            result["card_name"] = self._card_db.get_card_name(code) if code else ""
        # Resolve the prompt-level desc (yes/no, effect-yn) to display text.
        # desc == 0 means "no specific prompt string"; an unknown id means the
        # resolver can't find it; a template with placeholders we can't fill
        # also yields None. In all those cases the consumer falls back to a
        # synthesized question.
        if "desc" in result:
            desc = result["desc"]
            template = (
                self._resolver.resolve(desc)
                if (self._resolver and desc) else None
            )
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
        bytes_ = obs.actions[idx]
        msg_type = bytes_[0]
        category_byte = bytes_[1]
        code = decode_u32(bytes_, 2)
        controller = bytes_[6]
        location = bytes_[7]
        sequence = decode_u16(bytes_, 8)
        index_byte = bytes_[16]
        num_selected = bytes_[17]
        meta = obs.action_meta[idx] if idx < len(obs.action_meta) else None
        card_name = self._card_db.get_card_name(code) if code else ""

        description, category = self._dispatch(
            msg_type=msg_type,
            category_byte=category_byte,
            code=code,
            card_name=card_name,
            index_byte=index_byte,
            num_selected=num_selected,
            location=location,
            sequence=sequence,
            meta=meta,
        )

        return ActionDetails(
            index=idx,
            description=description,
            category=category,
            card_code=code,
            card_name=card_name,
            controller=controller,
            location=location,
            sequence=sequence,
            meta=meta,
        )

    def _dispatch(
        self, *, msg_type: int, category_byte: int, code: int,
        card_name: str, index_byte: int, num_selected: int,
        location: int, sequence: int, meta: ActionMeta | None,
    ) -> tuple[str, str]:
        cat = category_byte

        if msg_type == MSG_SELECT_IDLECMD:
            label, cat_str = _IDLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
            if code and card_name:
                base = f"{label} {card_name}"
                resolved = self._resolve_effect(meta)
                return (f"{base}: {resolved}" if resolved else base), cat_str
            return label, cat_str

        if msg_type == MSG_SELECT_BATTLECMD:
            label, cat_str = _BATTLE_DESCS.get(cat, (f"Action {cat}", "unknown"))
            if code and card_name:
                base = f"{label} {card_name}"
                resolved = self._resolve_effect(meta)
                return (f"{base}: {resolved}" if resolved else base), cat_str
            return label, cat_str

        if msg_type == MSG_SELECT_EFFECTYN:
            if cat == 0:
                base = f"Yes — activate {card_name}" if card_name else "Yes"
                resolved = self._resolve_effect(meta)
                return (f"{base}: {resolved}" if resolved else base), "yes"
            return "No", "no"

        if msg_type == MSG_SELECT_YESNO:
            if cat == 0:
                resolved = self._resolve_effect(meta)
                return (f"Yes — {resolved}" if resolved else "Yes"), "yes"
            return "No", "no"

        if msg_type == MSG_SELECT_OPTION:
            if meta is not None:
                resolved = (
                    self._resolver.resolve(meta.raw_value)
                    if (self._resolver and meta.raw_value is not None) else None
                )
                return (resolved or meta.label), "option"
            return f"Option {index_byte + 1}", "option"

        if msg_type == MSG_SELECT_CARD:
            if cat == 1:
                return (
                    f"Finish selecting ({num_selected} card{'s' if num_selected != 1 else ''})",
                    "finish",
                )
            label = f"Select {card_name}" if card_name else f"Select card #{index_byte}"
            return label, "select_card"

        if msg_type == MSG_SELECT_CHAIN:
            if cat == 1:
                return "Pass (no chain)", "pass"
            target = card_name or f"#{index_byte}"
            resolved = (
                self._resolver.resolve(meta.raw_value)
                if (self._resolver and meta is not None and meta.raw_value is not None)
                else None
            )
            if resolved and card_name:
                return f"Chain {target}: {resolved}", "chain"
            return f"Chain {target}", "chain"

        if msg_type in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
            zone = (
                "Monster" if location == LOCATION_MZONE
                else "Spell/Trap" if location == LOCATION_SZONE
                else "Unknown"
            )
            return f"Place in {zone} Zone {sequence + 1}", "place"

        if msg_type == MSG_SELECT_POSITION:
            pos_name = _POS_NAMES.get(index_byte, f"Position {index_byte}")
            desc = f"{card_name}: {pos_name}" if card_name else pos_name
            return desc, "position"

        if msg_type == MSG_SELECT_TRIBUTE:
            if cat == 1:
                return (
                    f"Finish tributing ({num_selected} card{'s' if num_selected != 1 else ''})",
                    "finish",
                )
            label = f"Tribute {card_name}" if card_name else f"Tribute card #{index_byte}"
            return label, "tribute"

        if msg_type == MSG_SELECT_UNSELECT_CARD:
            if cat == 1:
                return "Finish selection", "finish"
            label = f"Select {card_name}" if card_name else f"Select card #{index_byte}"
            return label, "select_card"

        if msg_type == MSG_ANNOUNCE_NUMBER:
            return (meta.label if meta else f"Announce #{index_byte}"), "number"

        if msg_type == MSG_ANNOUNCE_RACE:
            return (meta.label if meta else f"Race #{index_byte}"), "race"

        if msg_type == MSG_ANNOUNCE_ATTRIB:
            return (meta.label if meta else f"Attribute #{index_byte}"), "attribute"

        if msg_type == MSG_ROCK_PAPER_SCISSORS:
            return (meta.label if meta else f"RPS #{index_byte}"), "rps"

        if msg_type == MSG_SELECT_COUNTER:
            count = meta.extras["counter_count"] if meta else 0
            target = card_name or f"card #{index_byte}"
            return f"Remove {count} from {target}", "counter"

        # Fallback for sum, sort, etc.
        if card_name:
            return f"Select {card_name}", "select_card"
        return f"Action #{index_byte}", "unknown"

    def _resolve_effect(self, meta: ActionMeta | None) -> str | None:
        """Return resolved effect text for a kind=effect meta, or None."""
        if not self._resolver or meta is None or meta.kind != "effect":
            return None
        if meta.raw_value is None:
            return None
        return self._resolver.resolve(meta.raw_value)
