"""Project engine state into the structured observation the agent receives."""

from __future__ import annotations

from typing import Any

import numpy as np

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    POS_FACEUP,
    STATUS_DISABLED,
)
from yugioh_core.encoding import (
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
    ZONE_SLOTS,
    encode_chain_entry,
)
from yugioh_env.game_state import GameState
from yugioh_env.models import CardState, GlobalState


def build_observation(
    game_state: GameState,
    current_msg: dict | None,
    agent_player: int,
    query_fn=None,
    event_history: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the agent's observation from engine state.

    Args:
        game_state: Current GameState
        current_msg: The current SELECT message (if any)
        agent_player: Which player the agent controls (0 or 1)
        query_fn: Optional callable(player, location) -> list[dict] for querying cards

    Returns:
        Dict with the structured 'cards' (list[CardState]) and 'global_state'
        (GlobalState) carrying raw, unclamped engine values, plus the
        'pending_chain' and 'event_history' arrays.
    """
    opp_player = 1 - agent_player

    # Raw engine values throughout: clamping and masking belong to the encoder
    # that packs these into bytes, not here.
    global_state = GlobalState(
        my_lp=game_state.lp[agent_player],
        opp_lp=game_state.lp[opp_player],
        turn=game_state.turn_count,
        phase=game_state.phase,
        is_my_turn=game_state.current_player == agent_player,
        chain_count=game_state.chain_count,
        msg_type=(current_msg or {}).get("msg_type", 0),
        my_deck=game_state.deck_count[agent_player],
        my_hand=game_state.hand_count[agent_player],
        my_grave=game_state.grave_count[agent_player],
        my_banished=game_state.banished_count[agent_player],
        my_extra=game_state.extra_count[agent_player],
        opp_deck=game_state.deck_count[opp_player],
        opp_hand=game_state.hand_count[opp_player],
        opp_grave=game_state.grave_count[opp_player],
        opp_banished=game_state.banished_count[opp_player],
        opp_extra=game_state.extra_count[opp_player],
        is_finished=game_state.is_finished,
    )

    cards: list[CardState] = []

    # Fill card zones from query function if available
    if query_fn is not None:
        for player in [agent_player, opp_player]:
            is_agent = player == agent_player
            for loc, slot_name in [
                (LOCATION_HAND, "hand"),
                (LOCATION_MZONE, "mzone"),
                (LOCATION_SZONE, "szone"),
                (LOCATION_GRAVE, "grave"),
                (LOCATION_BANISHED, "banished"),
                (LOCATION_EXTRA, "extra"),
            ]:
                max_slots = ZONE_SLOTS[slot_name]
                queried = query_fn(player, loc)
                for i, cdata in enumerate(queried[:max_slots]):
                    # Filling stops silently once full, and every later zone
                    # then stops on its first card.
                    if len(cards) >= MAX_CARDS:
                        break

                    is_public = bool(cdata.get("is_public", 0))
                    is_hidden = bool(cdata.get("is_hidden", 0))
                    position = cdata.get("position", 0)
                    faceup = bool(position & POS_FACEUP) if position else False

                    # Determine visibility: agent sees own cards + public cards + face-up cards
                    visible = is_agent or is_public or faceup
                    if is_hidden:
                        visible = False

                    if visible:
                        cards.append(
                            CardState(
                                code=cdata.get("code", 0),
                                location=loc,
                                sequence=cdata.get("sequence", i),
                                position=position,
                                controller=0 if is_agent else 1,
                                is_public=is_public or faceup,
                                card_type=cdata.get("type", 0),
                                level=cdata.get("level", 0) or cdata.get("rank", 0),
                                attribute=cdata.get("attribute", 0),
                                race=cdata.get("race", 0),
                                attack=cdata.get("attack", 0),
                                defense=cdata.get("defense", 0),
                                lscale=cdata.get("lscale", 0),
                                rscale=cdata.get("rscale", 0),
                                link_marker=cdata.get("link_marker", 0),
                                counter_count=len(cdata.get("counters", [])),
                                negated=bool(cdata.get("status", 0) & STATUS_DISABLED),
                            )
                        )
                    else:
                        # Hidden card: only location/controller visible
                        cards.append(
                            CardState(
                                code=0,
                                location=loc,
                                sequence=cdata.get("sequence", i),
                                position=0,
                                controller=0 if is_agent else 1,
                                is_public=False,
                            )
                        )

    # Pending chain → (MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES) uint8 tensor.
    # Relativize controller (raw engine → 0=agent / 1=opponent) BEFORE encoding.
    pending_chain = np.zeros((MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES), dtype=np.uint8)
    for i, link in enumerate(game_state.pending_chain[:MAX_PENDING_CHAIN]):
        controller = 0 if link.controller == agent_player else 1
        pending_chain[i] = encode_chain_entry(
            code=link.code,
            desc=link.desc,
            controller=controller,
            location=link.location,
            sequence=link.sequence,
            chain_link=link.chain_link,
        )

    if event_history is None:
        event_history = np.zeros((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)

    return {
        "cards": cards,
        "global_state": global_state,
        "pending_chain": pending_chain,
        "event_history": event_history,
    }
