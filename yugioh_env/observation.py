"""Encode game state as numpy arrays for RL observation."""

from __future__ import annotations

import ctypes
import struct
import numpy as np

from yugioh_core.constants import (
    QUERY_BASIC,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    POS_FACEUP,
    QUERY_END,
    QUERY_RACE,
    QUERY_LINK,
    QUERY_OVERLAY_CARD,
    QUERY_COUNTERS,
)
from yugioh_core.encoding import (
    MAX_CARDS,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    ACTION_FEATURES,
    ZONE_SLOTS,
    encode_u16,
    encode_u32,
    encode_card,
)
from yugioh_env.game_state import GameState


def _parse_query_buffer(data: bytes) -> list[dict]:
    """Parse an OCG_DuelQueryLocation response buffer.

    Wire format (edo9300 ygopro-core):

        [0..3]    uint32  total_data_size       (bytes after this header)
        [4..]     per-slot loop until total_data exhausted:
                    empty slot:  int16(0)        — 2 bytes
                    card:        repeated field blocks until QUERY_END:
                                   uint16 field_size  (bytes for flag + value)
                                   uint32 flag
                                   bytes  value[field_size - 4]
                                 terminator: uint16(4) + uint32(QUERY_END)

    Empty slots only occur in MZONE/SZONE (fixed-size vector storage with
    nullptr holes); other zones never emit them. Each non-empty card dict
    carries a `sequence` key with the engine's slot index (counted across
    empty slots, so MZONE slot 5 with slots 0-4 empty has sequence=5).

    Raises:
        ValueError: An unknown flag was emitted. Indicates engine submodule
            drift or a missing handler in this parser; failing loudly is
            preferred to silently dropping the field, which would let the
            model train on incomplete observations.
        AssertionError: The parser's byte cursor doesn't land exactly on
            `total_size + 4` at function exit. Detects size-mismatch drift
            in the wire format.
    """
    if len(data) < 4:
        return []
    total_size = struct.unpack_from("<I", data, 0)[0]
    end = 4 + total_size
    cards: list[dict] = []
    pos = 4
    seq = 0

    while pos < end:
        first_u16 = struct.unpack_from("<H", data, pos)[0]
        if first_u16 == 0:
            cards.append({})
            pos += 2
            seq += 1
            continue

        card: dict = {"sequence": seq}
        while True:
            field_size = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            assert field_size >= 4, (
                f"malformed query buffer: field_size={field_size} < 4 "
                f"at pos={pos - 2}"
            )
            flag = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            data_size = field_size - 4

            if flag == QUERY_END:
                break

            value_start = pos
            if flag in _FLAG_TO_KEY_U32:
                assert data_size == 4, (
                    f"unexpected data_size={data_size} for u32 flag 0x{flag:x}"
                )
                card[_FLAG_TO_KEY_U32[flag]] = struct.unpack_from(
                    "<I", data, value_start,
                )[0]
            elif flag in _FLAG_TO_KEY_I32:
                assert data_size == 4, (
                    f"unexpected data_size={data_size} for i32 flag 0x{flag:x}"
                )
                card[_FLAG_TO_KEY_I32[flag]] = struct.unpack_from(
                    "<i", data, value_start,
                )[0]
            elif flag in _FLAG_TO_KEY_U8:
                assert data_size == 1, (
                    f"unexpected data_size={data_size} for u8 flag 0x{flag:x}"
                )
                card[_FLAG_TO_KEY_U8[flag]] = data[value_start]
            elif flag == QUERY_RACE:
                assert data_size == 8, (
                    f"unexpected data_size={data_size} for QUERY_RACE"
                )
                card["race"] = struct.unpack_from("<Q", data, value_start)[0]
            elif flag == QUERY_LINK:
                assert data_size == 8, (
                    f"unexpected data_size={data_size} for QUERY_LINK"
                )
                card["link_rating"] = struct.unpack_from(
                    "<I", data, value_start,
                )[0]
                card["link_marker"] = struct.unpack_from(
                    "<I", data, value_start + 4,
                )[0]
            elif flag == QUERY_OVERLAY_CARD:
                # variable length: u32 count + N×u32
                count = struct.unpack_from("<I", data, value_start)[0]
                assert data_size == 4 + 4 * count, (
                    f"QUERY_OVERLAY_CARD data_size mismatch: "
                    f"got {data_size}, expected {4 + 4 * count}"
                )
                card["overlay_cards"] = [
                    struct.unpack_from("<I", data, value_start + 4 + j * 4)[0]
                    for j in range(count)
                ]
            elif flag == QUERY_COUNTERS:
                count = struct.unpack_from("<I", data, value_start)[0]
                assert data_size == 4 + 4 * count, (
                    f"QUERY_COUNTERS data_size mismatch: "
                    f"got {data_size}, expected {4 + 4 * count}"
                )
                card["counters"] = [
                    struct.unpack_from("<I", data, value_start + 4 + j * 4)[0]
                    for j in range(count)
                ]
            else:
                raise ValueError(
                    f"unknown query flag 0x{flag:x} at pos={value_start - 6}"
                )
            pos += data_size

        cards.append(card)
        seq += 1

    assert pos == end, (
        f"query buffer parse drift: pos={pos} but end={end} "
        f"(total_size={total_size})"
    )
    return cards


# Flag → key tables for `_parse_query_buffer`. Keep in sync with the
# canonical table in docs/superpowers/specs/2026-05-08-query-buffer-parser-fix-design.md
# and with `board_state.py`'s parser until they're deduplicated.

_FLAG_TO_KEY_U32 = {
    0x1: "code",            # QUERY_CODE
    0x2: "position",        # QUERY_POSITION
    0x4: "alias",           # QUERY_ALIAS
    0x8: "type",            # QUERY_TYPE
    0x10: "level",          # QUERY_LEVEL
    0x20: "rank",           # QUERY_RANK
    0x40: "attribute",      # QUERY_ATTRIBUTE
    0x1000: "reason",       # QUERY_REASON
    0x80000: "status",      # QUERY_STATUS
    0x200000: "lscale",     # QUERY_LSCALE
    0x400000: "rscale",     # QUERY_RSCALE
    0x2000000: "cover",     # QUERY_COVER
}

_FLAG_TO_KEY_I32 = {
    0x100: "attack",        # QUERY_ATTACK (int32, can be negative)
    0x200: "defense",       # QUERY_DEFENSE
    0x400: "base_attack",   # QUERY_BASE_ATTACK
    0x800: "base_defense",  # QUERY_BASE_DEFENSE
}

_FLAG_TO_KEY_U8 = {
    0x40000: "owner",       # QUERY_OWNER
    0x100000: "is_public",  # QUERY_IS_PUBLIC
    0x1000000: "is_hidden", # QUERY_IS_HIDDEN
}


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
    global_state[idx], global_state[idx + 1] = encode_u16(lp)
    idx += 2
    # opp_lp (2 bytes)
    lp = min(game_state.lp[opp_player], 65535)
    global_state[idx], global_state[idx + 1] = encode_u16(lp)
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
                            sequence=cdata.get("sequence", i),
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
                            sequence=cdata.get("sequence", i),
                            position=0,
                            controller=0 if is_agent else 1,
                            is_public=False,
                        )
                    card_idx += 1

    return {
        "cards": cards,
        "global_state": global_state,
    }
