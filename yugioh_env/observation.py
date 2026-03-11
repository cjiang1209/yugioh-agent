"""Encode game state as numpy arrays for RL observation."""

from __future__ import annotations

import ctypes
import struct
import numpy as np

from yugioh_env.constants import (
    QUERY_BASIC,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    POS_FACEUP,
    QUERY_END,
)
from yugioh_env.game_state import GameState

# ─── Observation dimensions ──────────────────────────────────────────────────
MAX_CARDS = 200
CARD_FEATURES = 42
GLOBAL_FEATURES = 20
MAX_ACTIONS = 32
ACTION_FEATURES = 12

# Zone slot allocations per player
ZONE_SLOTS = {
    "hand": 15,
    "mzone": 7,
    "szone": 6,
    "grave": 30,
    "banished": 20,
    "extra": 15,
}
# Total per player = 15+7+6+30+20+15 = 93, times 2 = 186, leaves room for overflow


def _encode_u16(val: int) -> tuple[int, int]:
    """Encode a uint16 value as two uint8 bytes (little-endian)."""
    return val & 0xFF, (val >> 8) & 0xFF


def _encode_i16_clamped(val: int) -> tuple[int, int]:
    """Encode a potentially large int as clamped uint16 (0-65535)."""
    val = max(0, min(65535, val))
    return val & 0xFF, (val >> 8) & 0xFF


def _encode_u32(val: int) -> tuple[int, int, int, int]:
    """Encode a uint32 value as four uint8 bytes (little-endian)."""
    return val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF


def encode_card(
    code: int,
    location: int,
    sequence: int,
    position: int,
    controller: int,
    is_public: bool,
    card_type: int = 0,
    level: int = 0,
    attribute: int = 0,
    race: int = 0,
    attack: int = 0,
    defense: int = 0,
    lscale: int = 0,
    rscale: int = 0,
    link_marker: int = 0,
    counter_count: int = 0,
    negated: bool = False,
    is_overlay: bool = False,
) -> np.ndarray:
    """Encode a single card as a feature vector.

    Returns:
        np.ndarray of shape (CARD_FEATURES,) dtype uint8
    """
    feat = np.zeros(CARD_FEATURES, dtype=np.uint8)
    idx = 0

    # card_id (4 bytes, uint32 LE)
    feat[idx], feat[idx + 1], feat[idx + 2], feat[idx + 3] = _encode_u32(code & 0xFFFFFFFF)
    idx += 4

    # location, sequence, position, controller, is_public
    feat[idx] = location & 0xFF
    idx += 1
    feat[idx] = min(sequence, 255)
    idx += 1
    feat[idx] = position & 0xFF
    idx += 1
    feat[idx] = controller & 0xFF
    idx += 1
    feat[idx] = 1 if is_public else 0
    idx += 1

    # type (4 bytes)
    feat[idx] = card_type & 0xFF
    feat[idx + 1] = (card_type >> 8) & 0xFF
    feat[idx + 2] = (card_type >> 16) & 0xFF
    feat[idx + 3] = (card_type >> 24) & 0xFF
    idx += 4

    # level
    feat[idx] = min(level, 255)
    idx += 1

    # attribute
    feat[idx] = attribute & 0xFF
    idx += 1

    # race (4 bytes, uint32 LE)
    feat[idx], feat[idx + 1], feat[idx + 2], feat[idx + 3] = _encode_u32(race & 0xFFFFFFFF)
    idx += 4

    # ATK (2 bytes, clamped)
    feat[idx], feat[idx + 1] = _encode_i16_clamped(attack if attack >= 0 else 0)
    idx += 2

    # DEF (2 bytes, clamped)
    feat[idx], feat[idx + 1] = _encode_i16_clamped(defense if defense >= 0 else 0)
    idx += 2

    # lscale, rscale
    feat[idx] = min(lscale, 255)
    idx += 1
    feat[idx] = min(rscale, 255)
    idx += 1

    # link_marker (2 bytes)
    feat[idx], feat[idx + 1] = _encode_u16(link_marker)
    idx += 2

    # counter_count
    feat[idx] = min(counter_count, 255)
    idx += 1

    # negated
    feat[idx] = 1 if negated else 0
    idx += 1

    # is_overlay
    feat[idx] = 1 if is_overlay else 0
    idx += 1

    # Remaining features are padding (zero)
    return feat


def _parse_query_buffer(data: bytes) -> list[dict]:
    """Parse a query location/field buffer into card data dicts.

    Query format (edo9300): Each card starts with a 4-byte total_size.
    Then a sequence of flag(4) + data blocks until QUERY_END.
    """
    cards = []
    pos = 0
    while pos < len(data) - 4:
        total_size = struct.unpack_from("<I", data, pos)[0]
        if total_size == 0:
            break
        end = pos + total_size
        pos += 4  # skip total_size

        card: dict = {}
        while pos < end - 4:
            flag = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            if flag == QUERY_END:
                break

            if flag == 0x1:  # QUERY_CODE
                card["code"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x2:  # QUERY_POSITION
                card["position"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x4:  # QUERY_ALIAS
                card["alias"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x8:  # QUERY_TYPE
                card["type"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x10:  # QUERY_LEVEL
                card["level"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x20:  # QUERY_RANK
                card["rank"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x40:  # QUERY_ATTRIBUTE
                card["attribute"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x80:  # QUERY_RACE
                card["race"] = struct.unpack_from("<Q", data, pos)[0]
                pos += 8
            elif flag == 0x100:  # QUERY_ATTACK
                card["attack"] = struct.unpack_from("<i", data, pos)[0]
                pos += 4
            elif flag == 0x200:  # QUERY_DEFENSE
                card["defense"] = struct.unpack_from("<i", data, pos)[0]
                pos += 4
            elif flag == 0x400:  # QUERY_BASE_ATTACK
                card["base_attack"] = struct.unpack_from("<i", data, pos)[0]
                pos += 4
            elif flag == 0x800:  # QUERY_BASE_DEFENSE
                card["base_defense"] = struct.unpack_from("<i", data, pos)[0]
                pos += 4
            elif flag == 0x1000:  # QUERY_REASON
                card["reason"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x40000:  # QUERY_OWNER
                card["owner"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x80000:  # QUERY_STATUS
                card["status"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x100000:  # QUERY_IS_PUBLIC
                card["is_public"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x200000:  # QUERY_LSCALE
                card["lscale"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x400000:  # QUERY_RSCALE
                card["rscale"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x800000:  # QUERY_LINK
                card["link_rating"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                card["link_marker"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x1000000:  # QUERY_IS_HIDDEN
                card["is_hidden"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x2000000:  # QUERY_COVER
                card["cover"] = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            elif flag == 0x10000:  # QUERY_OVERLAY_CARD
                ov_count = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                overlays = []
                for _ in range(ov_count):
                    overlays.append(struct.unpack_from("<I", data, pos)[0])
                    pos += 4
                card["overlay_cards"] = overlays
            elif flag == 0x20000:  # QUERY_COUNTERS
                ct_count = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                counters = []
                for _ in range(ct_count):
                    counters.append(struct.unpack_from("<I", data, pos)[0])
                    pos += 4
                card["counters"] = counters
            else:
                # Unknown query flag, skip to next card
                break

        cards.append(card)
        pos = end

    return cards


def build_observation(
    game_state: GameState,
    current_msg: dict | None,
    agent_player: int,
    query_fn=None,
) -> dict[str, np.ndarray]:
    """Build the complete observation arrays.

    Args:
        game_state: Current GameState
        current_msg: The current SELECT message (if any)
        agent_player: Which player the agent controls (0 or 1)
        query_fn: Optional callable(player, location) -> list[dict] for querying cards

    Returns:
        Dict with 'cards', 'global_state' numpy arrays.
    """
    cards = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
    global_state = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)

    opp_player = 1 - agent_player

    # Fill global state
    idx = 0
    # my_lp (2 bytes)
    lp = min(game_state.lp[agent_player], 65535)
    global_state[idx], global_state[idx + 1] = _encode_u16(lp)
    idx += 2
    # opp_lp (2 bytes)
    lp = min(game_state.lp[opp_player], 65535)
    global_state[idx], global_state[idx + 1] = _encode_u16(lp)
    idx += 2
    # turn_count
    global_state[idx] = min(game_state.turn_count, 255)
    idx += 1
    # phase
    global_state[idx] = game_state.phase & 0xFF
    idx += 1
    # is_my_turn
    global_state[idx] = 1 if game_state.current_player == agent_player else 0
    idx += 1
    # chain_count
    global_state[idx] = min(game_state.chain_count, 255)
    idx += 1
    # msg_type
    global_state[idx] = (current_msg or {}).get("msg_type", 0) & 0xFF
    idx += 1
    # deck/hand/gy/banished/extra counts per player
    for p in [agent_player, opp_player]:
        global_state[idx] = min(game_state.deck_count[p], 255)
        idx += 1
        global_state[idx] = min(game_state.hand_count[p], 255)
        idx += 1
        global_state[idx] = min(game_state.grave_count[p], 255)
        idx += 1
        global_state[idx] = min(game_state.banished_count[p], 255)
        idx += 1
        global_state[idx] = min(game_state.extra_count[p], 255)
        idx += 1
    # is_finished
    global_state[idx] = 1 if game_state.is_finished else 0
    idx += 1

    # Fill card zones from query function if available
    if query_fn is not None:
        card_idx = 0

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
                    if card_idx >= MAX_CARDS:
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
                        cards[card_idx] = encode_card(
                            code=cdata.get("code", 0),
                            location=loc,
                            sequence=i,
                            position=position,
                            controller=0 if is_agent else 1,
                            is_public=is_public or faceup,
                            card_type=cdata.get("type", 0),
                            level=cdata.get("level", 0) or cdata.get("rank", 0),
                            attribute=cdata.get("attribute", 0),
                            race=cdata.get("race", 0) & 0xFFFFFFFF,
                            attack=cdata.get("attack", 0),
                            defense=cdata.get("defense", 0),
                            lscale=cdata.get("lscale", 0),
                            rscale=cdata.get("rscale", 0),
                            link_marker=cdata.get("link_marker", 0),
                            counter_count=len(cdata.get("counters", [])),
                            negated=bool(cdata.get("status", 0) & 0x1),
                        )
                    else:
                        # Hidden card: only location/controller visible
                        cards[card_idx] = encode_card(
                            code=0,
                            location=loc,
                            sequence=i,
                            position=0,
                            controller=0 if is_agent else 1,
                            is_public=False,
                        )
                    card_idx += 1

    return {
        "cards": cards,
        "global_state": global_state,
    }
