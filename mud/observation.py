"""Build RL-compatible observations from MUD game state.

Converts ``MUDGameState`` + ``ParsedPrompt`` into numpy arrays matching
the training environment's observation format.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

from core.card_database import CardDatabase
from core.constants import (
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_ONFIELD,
    LOCATION_SZONE,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_COUNTER,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    PHASE_BATTLE,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_DRAW,
    PHASE_END,
    PHASE_MAIN1,
    PHASE_MAIN2,
    PHASE_STANDBY,
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    encode_action,
    encode_card,
    encode_global,
)

from .cmd_handler import parse_cardspec
from .game_state import CardEntry, MUDGameState
from .text_parser import ParsedPrompt, PromptType

_EFFECTYN_NAME_RE = re.compile(r"Do you want to use the effect from (.+?)(?:\s+in\s+[a-z]+\d+)?\?")


# ---------------------------------------------------------------------------
# Phase string → RL bitmask
# ---------------------------------------------------------------------------

PHASE_MAP: dict[str, int] = {
    "draw phase": PHASE_DRAW,
    "standby phase": PHASE_STANDBY,
    "main1 phase": PHASE_MAIN1,
    "main phase 1": PHASE_MAIN1,
    "battle phase": PHASE_BATTLE,
    "battle start phase": PHASE_BATTLE_START,
    "battle step phase": PHASE_BATTLE_STEP,
    "damage phase": PHASE_DAMAGE,
    "damage calculation phase": PHASE_DAMAGE_CAL,
    "main2 phase": PHASE_MAIN2,
    "main phase 2": PHASE_MAIN2,
    "end phase": PHASE_END,
}

# ---------------------------------------------------------------------------
# Position string → RL bitmask
# ---------------------------------------------------------------------------

POSITION_MAP: dict[str, int] = {
    "face-up attack": POS_FACEUP_ATTACK,
    "face-down attack": POS_FACEDOWN_ATTACK,
    "face-up defense": POS_FACEUP_DEFENSE,
    "face-down defense": POS_FACEDOWN_DEFENSE,
    # The MUD server's text for a face-up spell/trap, and for a face-up extra
    # deck card.
    "face-up": POS_FACEUP,
    # The MUD server's text for a set spell/trap, and for a face-down extra
    # deck card.
    "face down": POS_FACEDOWN,
}

# The in-process ygopro-core reports extra deck cards in defense position.
_EXTRA_POSITION = {POS_FACEUP: POS_FACEUP_DEFENSE, POS_FACEDOWN: POS_FACEDOWN_DEFENSE}


def _sequences(zone: list[CardEntry], location: int) -> list[int]:
    """The sequence of each card in a zone.

    Outside the field it is the card's place in the zone. On the field it is
    the slot the card occupies: a card's MUD spec ("m3", or "om3" for an
    opponent's card) is its slot plus one. The server's summon text names no
    zone, so a card summoned since the last board re-read has no spec; it
    takes the lowest slot no other card claims.
    """
    if not location & LOCATION_ONFIELD:
        return list(range(len(zone)))
    slots: list[int | None] = []
    for entry in zone:
        spec_location, sequence = parse_cardspec(entry.spec)
        slots.append(sequence if spec_location == location else None)
    free = (s for s in itertools.count() if s not in slots)
    return [s if s is not None else next(free) for s in slots]


# ---------------------------------------------------------------------------
# PromptType → MSG_SELECT_* mapping
# ---------------------------------------------------------------------------

PROMPT_MSG_MAP: dict[PromptType, int] = {
    PromptType.IDLE_CMD: MSG_SELECT_IDLECMD,
    PromptType.BATTLE_MENU: MSG_SELECT_BATTLECMD,
    PromptType.SELECT_EFFECTYN: MSG_SELECT_EFFECTYN,
    PromptType.SELECT_YESNO: MSG_SELECT_YESNO,
    PromptType.SELECT_OPTION: MSG_SELECT_OPTION,
    PromptType.SELECT_CARD: MSG_SELECT_CARD,
    PromptType.SELECT_CHAIN: MSG_SELECT_CHAIN,
    PromptType.SELECT_PLACE: MSG_SELECT_PLACE,
    PromptType.SELECT_POSITION: MSG_SELECT_POSITION,
    PromptType.SELECT_TRIBUTE: MSG_SELECT_TRIBUTE,
    PromptType.SELECT_COUNTER: MSG_SELECT_COUNTER,
    PromptType.SELECT_SUM: MSG_SELECT_SUM,
    PromptType.SELECT_UNSELECT: MSG_SELECT_UNSELECT_CARD,
}


# ---------------------------------------------------------------------------
# MUDObservationBuilder
# ---------------------------------------------------------------------------


class MUDObservationBuilder:
    """Convert MUD game state + prompt into RL observation arrays."""

    def __init__(self, card_db: CardDatabase) -> None:
        self._card_db = card_db

    def build(
        self,
        game_state: MUDGameState,
        prompt: ParsedPrompt,
    ) -> dict[str, np.ndarray]:
        """Build observation dict matching RL training format.

        Returns:
            Dict with keys: ``cards`` (200,42), ``global_state`` (21,),
            ``actions`` (32,12), ``action_mask`` (32,).
        """
        cards = self._build_cards(game_state)
        global_state = self._build_global_state(game_state, prompt)
        actions, action_mask = self._build_actions(game_state, prompt)
        return {
            "cards": cards,
            "global_state": global_state,
            "actions": actions,
            "action_mask": action_mask,
        }

    # ------------------------------------------------------------------
    # Card encoding
    # ------------------------------------------------------------------

    def _build_cards(self, gs: MUDGameState) -> np.ndarray:
        rows: list[np.ndarray] = []

        # Agent zones (controller=0)
        for loc, zone in [
            (LOCATION_HAND, gs.my_hand),
            (LOCATION_MZONE, gs.my_mzone),
            (LOCATION_SZONE, gs.my_szone),
            (LOCATION_GRAVE, gs.my_graveyard),
            (LOCATION_BANISHED, gs.my_banished),
            (LOCATION_EXTRA, gs.my_extra),
        ]:
            for seq, entry in zip(_sequences(zone, loc), zone, strict=True):
                rows.append(self._encode_entry(entry, loc, seq, controller=0, is_public=True))

        # Opponent hand: only its size is known
        for i in range(gs.opp_hand_count):
            rows.append(
                encode_card(
                    code=0,
                    location=LOCATION_HAND,
                    sequence=i,
                    position=0,
                    controller=1,
                    is_public=False,
                )
            )

        # Opponent zones (controller=1)
        for loc, zone in [
            (LOCATION_MZONE, gs.opp_mzone),
            (LOCATION_SZONE, gs.opp_szone),
            (LOCATION_GRAVE, gs.opp_graveyard),
            (LOCATION_BANISHED, gs.opp_banished),
            (LOCATION_EXTRA, gs.opp_extra),
        ]:
            for seq, entry in zip(_sequences(zone, loc), zone, strict=True):
                is_faceup = self._is_faceup(entry)
                is_public = is_faceup and entry.code != 0
                rows.append(self._encode_entry(entry, loc, seq, controller=1, is_public=is_public))

        cards = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        rows = rows[:MAX_CARDS]
        if rows:
            cards[: len(rows)] = np.stack(rows)
        return cards

    def _encode_entry(
        self,
        entry: CardEntry,
        location: int,
        sequence: int,
        controller: int,
        is_public: bool,
    ) -> np.ndarray:
        position = POSITION_MAP.get(entry.position, 0)
        if location == LOCATION_EXTRA:
            position = _EXTRA_POSITION.get(position, position)

        if not is_public or entry.code == 0:
            return encode_card(
                code=0,
                location=location,
                sequence=sequence,
                position=position if is_public or location & LOCATION_ONFIELD else 0,
                controller=controller,
                is_public=False,
            )

        card_data = self._card_db.get_card(entry.code)
        if card_data is None:
            return encode_card(
                code=entry.code,
                location=location,
                sequence=sequence,
                position=position,
                controller=controller,
                is_public=True,
            )

        return encode_card(
            code=entry.code,
            location=location,
            sequence=sequence,
            position=position,
            controller=controller,
            is_public=True,
            card_type=card_data.get("type", 0),
            level=card_data.get("level", 0),
            attribute=card_data.get("attribute", 0),
            race=card_data.get("race", 0) & 0xFFFFFFFF,
            attack=card_data.get("attack", 0),
            defense=card_data.get("defense", 0),
            lscale=card_data.get("lscale", 0),
            rscale=card_data.get("rscale", 0),
            link_marker=card_data.get("link_marker", 0),
        )

    @staticmethod
    def _is_faceup(entry: CardEntry) -> bool:
        pos = entry.position.lower()
        return "face-down" not in pos and "face down" not in pos

    # ------------------------------------------------------------------
    # Global state encoding
    # ------------------------------------------------------------------

    def _build_global_state(
        self,
        gs: MUDGameState,
        prompt: ParsedPrompt,
    ) -> np.ndarray:
        return encode_global(
            gs.my_lp,
            gs.opp_lp,
            gs.turn,
            PHASE_MAP.get(gs.phase.lower(), 0),
            gs.is_my_turn,
            # chain_count -- the MUD text parser doesn't track chain depth yet.
            0,
            PROMPT_MSG_MAP.get(prompt.prompt_type, 0),
            gs.my_deck_count,
            len(gs.my_hand),
            len(gs.my_graveyard),
            len(gs.my_banished),
            len(gs.my_extra),
            gs.opp_deck_count,
            gs.opp_hand_count,
            len(gs.opp_graveyard),
            len(gs.opp_banished),
            len(gs.opp_extra),
            # is_finished -- we're still playing if we're building an observation.
            False,
        )

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------

    def _build_actions(
        self,
        gs: MUDGameState,
        prompt: ParsedPrompt,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)

        msg_type = PROMPT_MSG_MAP.get(prompt.prompt_type, 0)

        if prompt.prompt_type in (PromptType.IDLE_CMD, PromptType.BATTLE_MENU):
            self._encode_structured_actions(actions, mask, prompt, msg_type)
        elif prompt.prompt_type in (PromptType.SELECT_EFFECTYN, PromptType.SELECT_YESNO):
            self._encode_binary_choice(actions, mask, msg_type, prompt, gs)
        else:
            self._encode_indexed_options(actions, mask, prompt, msg_type)

        return actions, mask

    def _encode_structured_actions(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        prompt: ParsedPrompt,
        msg_type: int,
    ) -> None:
        # ``index`` is the position within the sub-category list, not the flat
        # action list, which is what the in-process encoder puts there.
        #
        # controller is 0 because `_resolve_card_code` looks only in the
        # agent's own zones and no cardspec prefix names a side; a battle row
        # names the attacker, not its target. A future opponent-addressable
        # cardspec has to revisit this.
        cat_counts: dict[int, int] = {}
        for i, sa in enumerate(prompt.structured_actions[:MAX_ACTIONS]):
            cat = sa.category
            sub_idx = cat_counts.get(cat, 0)
            cat_counts[cat] = sub_idx + 1
            actions[i] = encode_action(
                msg_type,
                category=cat,
                code=sa.card_code,
                location=sa.location,
                sequence=sa.sequence,
                index=sub_idx,
            )
            mask[i] = 1

    def _encode_binary_choice(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        msg_type: int,
        prompt: ParsedPrompt,
        gs: MUDGameState,
    ) -> None:
        # RL encoding: Yes → category=0, No → category=1, both index=0.
        # For EFFECTYN, include the card code extracted from the prompt text.
        code = 0
        if prompt.prompt_type == PromptType.SELECT_EFFECTYN and prompt.raw_lines:
            m = _EFFECTYN_NAME_RE.match(prompt.raw_lines[0])
            if m:
                code = gs.resolve_code(m.group(1))
        for i in range(2):
            # Both choices carry index 0; the category is the answer.
            actions[i] = encode_action(msg_type, category=i, code=code)
            mask[i] = 1

    def _encode_indexed_options(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        prompt: ParsedPrompt,
        msg_type: int,
    ) -> None:
        """Every prompt the parser reads only as a numbered list.

        The row carries the option's number and nothing else, so one option
        is offered even when the text listed none -- a prompt with an empty
        mask is one the agent cannot answer at all.
        """
        n = max(len(prompt.options), 1)
        for i in range(min(n, MAX_ACTIONS)):
            actions[i] = encode_action(msg_type, index=i)
            mask[i] = 1
