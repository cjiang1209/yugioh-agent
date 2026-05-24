"""Build RL-compatible observations from MUD game state.

Converts ``MUDGameState`` + ``ParsedPrompt`` into numpy arrays matching
the training environment's observation format.
"""

from __future__ import annotations

import re

import numpy as np

from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
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
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    ZONE_SLOTS,
    encode_card,
    encode_u16,
    encode_u32,
)
from yugioh_mud.game_state import CardEntry, MUDGameState
from yugioh_mud.text_parser import ParsedPrompt, PromptType

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
    "face-up": POS_FACEUP_ATTACK,
    "face down": POS_FACEDOWN_DEFENSE,
}

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
            Dict with keys: ``cards`` (200,42), ``global_state`` (20,),
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
        cards = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        idx = 0

        # Agent zones (controller=0)
        for loc, zone, slot_name in [
            (LOCATION_HAND, gs.my_hand, "hand"),
            (LOCATION_MZONE, gs.my_mzone, "mzone"),
            (LOCATION_SZONE, gs.my_szone, "szone"),
            (LOCATION_GRAVE, gs.my_graveyard, "grave"),
            (LOCATION_BANISHED, gs.my_banished, "banished"),
            (LOCATION_EXTRA, gs.my_extra, "extra"),
        ]:
            max_slots = ZONE_SLOTS[slot_name]
            for i, entry in enumerate(zone[:max_slots]):
                if idx >= MAX_CARDS:
                    break
                cards[idx] = self._encode_entry(entry, loc, i, controller=0, is_public=True)
                idx += 1

        # Opponent zones (controller=1)
        for loc, zone, slot_name in [
            (LOCATION_HAND, None, "hand"),  # special: count-only
            (LOCATION_MZONE, gs.opp_mzone, "mzone"),
            (LOCATION_SZONE, gs.opp_szone, "szone"),
            (LOCATION_GRAVE, gs.opp_graveyard, "grave"),
            (LOCATION_BANISHED, gs.opp_banished, "banished"),
            (LOCATION_EXTRA, gs.opp_extra, "extra"),
        ]:
            max_slots = ZONE_SLOTS[slot_name]
            if zone is None:
                # Opponent hand — hidden entries
                for i in range(min(gs.opp_hand_count, max_slots)):
                    if idx >= MAX_CARDS:
                        break
                    cards[idx] = encode_card(
                        code=0,
                        location=LOCATION_HAND,
                        sequence=i,
                        position=0,
                        controller=1,
                        is_public=False,
                    )
                    idx += 1
                continue

            for i, entry in enumerate(zone[:max_slots]):
                if idx >= MAX_CARDS:
                    break
                is_faceup = self._is_faceup(entry)
                is_public = is_faceup and entry.code != 0
                cards[idx] = self._encode_entry(entry, loc, i, controller=1, is_public=is_public)
                idx += 1

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

        if not is_public or entry.code == 0:
            return encode_card(
                code=0,
                location=location,
                sequence=sequence,
                position=0 if not is_public else position,
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
        g = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
        idx = 0

        # my_lp (2 bytes)
        g[idx], g[idx + 1] = encode_u16(min(gs.my_lp, 65535))
        idx += 2
        # opp_lp (2 bytes)
        g[idx], g[idx + 1] = encode_u16(min(gs.opp_lp, 65535))
        idx += 2
        # turn_count
        g[idx] = min(gs.turn, 255)
        idx += 1
        # phase
        g[idx] = PHASE_MAP.get(gs.phase.lower(), 0) & 0xFF
        idx += 1
        # is_my_turn
        g[idx] = 1 if gs.is_my_turn else 0
        idx += 1
        # chain_count — TODO: MUD text parser doesn't track chain depth yet
        g[idx] = 0
        idx += 1
        # msg_type
        g[idx] = PROMPT_MSG_MAP.get(prompt.prompt_type, 0) & 0xFF
        idx += 1
        # Per-player counts: [agent, opponent] × [deck, hand, grave, banished, extra]
        g[idx] = min(gs.my_deck_count, 255)
        idx += 1
        g[idx] = min(len(gs.my_hand), 255)
        idx += 1
        g[idx] = min(len(gs.my_graveyard), 255)
        idx += 1
        g[idx] = min(len(gs.my_banished), 255)
        idx += 1
        g[idx] = min(len(gs.my_extra), 255)
        idx += 1
        g[idx] = min(gs.opp_deck_count, 255)
        idx += 1
        g[idx] = min(gs.opp_hand_count, 255)
        idx += 1
        g[idx] = min(len(gs.opp_graveyard), 255)
        idx += 1
        g[idx] = min(len(gs.opp_banished), 255)
        idx += 1
        g[idx] = min(len(gs.opp_extra), 255)
        idx += 1
        # is_finished — always 0 (we're still playing if we're building obs)
        g[idx] = 0
        idx += 1

        return g

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
        elif prompt.prompt_type in (PromptType.SELECT_CARD, PromptType.SELECT_TRIBUTE):
            self._encode_option_actions(actions, mask, prompt, msg_type, gs)
        else:
            self._encode_generic_options(actions, mask, prompt, msg_type)

        return actions, mask

    def _encode_structured_actions(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        prompt: ParsedPrompt,
        msg_type: int,
    ) -> None:
        # Track per-category counts so ``index`` matches the RL encoding
        # (index within the sub-category list, not the flat action list).
        cat_counts: dict[int, int] = {}
        for i, sa in enumerate(prompt.structured_actions[:MAX_ACTIONS]):
            cat = sa.category
            sub_idx = cat_counts.get(cat, 0)
            cat_counts[cat] = sub_idx + 1

            feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
            feat[0] = msg_type & 0xFF
            feat[1] = cat & 0xFF
            code = sa.card_code
            feat[2], feat[3], feat[4], feat[5] = encode_u32(code)
            feat[6] = sa.location & 0xFF
            feat[7] = min(sa.sequence, 255)
            feat[8] = sub_idx & 0xFF
            feat[9] = 1  # num_selected
            actions[i] = feat
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
            feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
            feat[0] = msg_type & 0xFF
            feat[1] = i & 0xFF  # category: 0=Yes, 1=No
            feat[2], feat[3], feat[4], feat[5] = encode_u32(code)
            # feat[8] = 0 (index stays 0 for both)
            actions[i] = feat
            mask[i] = 1

    def _encode_option_actions(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        prompt: ParsedPrompt,
        msg_type: int,
        gs: MUDGameState,
    ) -> None:
        for i, _opt in enumerate(prompt.options[:MAX_ACTIONS]):
            feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
            feat[0] = msg_type & 0xFF
            feat[8] = i & 0xFF
            feat[9] = 1
            actions[i] = feat
            mask[i] = 1

    def _encode_generic_options(
        self,
        actions: np.ndarray,
        mask: np.ndarray,
        prompt: ParsedPrompt,
        msg_type: int,
    ) -> None:
        n = max(len(prompt.options), 1)
        for i in range(min(n, MAX_ACTIONS)):
            feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
            feat[0] = msg_type & 0xFF
            feat[8] = i & 0xFF
            actions[i] = feat
            mask[i] = 1
