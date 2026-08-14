"""Decoder for OCG_DuelQueryLocation response buffers.

Single source of truth used by both the RL board observation path
(yugioh_env/observation.py via Duel.query_location) and the web UI's
board-state builder (yugioh_env/server/board_state.py). Strict
semantics: raises on unknown flags, asserts data-size invariants,
asserts the byte cursor lands exactly at the end of the declared
buffer. This catches engine-submodule drift loudly instead of
silently dropping fields or desynchronizing the cursor.
"""

from __future__ import annotations

import struct

from yugioh_core.constants import (
    QUERY_ALIAS,
    QUERY_ATTACK,
    QUERY_ATTRIBUTE,
    QUERY_BASE_ATTACK,
    QUERY_BASE_DEFENSE,
    QUERY_CODE,
    QUERY_COUNTERS,
    QUERY_COVER,
    QUERY_DEFENSE,
    QUERY_END,
    QUERY_IS_HIDDEN,
    QUERY_IS_PUBLIC,
    QUERY_LEVEL,
    QUERY_LINK,
    QUERY_LSCALE,
    QUERY_OVERLAY_CARD,
    QUERY_OWNER,
    QUERY_POSITION,
    QUERY_RACE,
    QUERY_RANK,
    QUERY_REASON,
    QUERY_RSCALE,
    QUERY_STATUS,
    QUERY_TYPE,
)

# Flag → key tables for `parse_query_location`. Three families by
# value width (u32, i32, u8); QUERY_RACE / QUERY_LINK / QUERY_OVERLAY_CARD
# / QUERY_COUNTERS have non-table-driven shapes and are dispatched inline.

_FLAG_TO_KEY_U32 = {
    QUERY_CODE: "code",
    QUERY_POSITION: "position",
    QUERY_ALIAS: "alias",
    QUERY_TYPE: "type",
    QUERY_LEVEL: "level",
    QUERY_RANK: "rank",
    QUERY_ATTRIBUTE: "attribute",
    QUERY_REASON: "reason",
    QUERY_STATUS: "status",
    QUERY_LSCALE: "lscale",
    QUERY_RSCALE: "rscale",
    QUERY_COVER: "cover",
}

_FLAG_TO_KEY_I32 = {
    QUERY_ATTACK: "attack",  # can be negative
    QUERY_DEFENSE: "defense",
    QUERY_BASE_ATTACK: "base_attack",
    QUERY_BASE_DEFENSE: "base_defense",
}

_FLAG_TO_KEY_U8 = {
    QUERY_OWNER: "owner",
    QUERY_IS_PUBLIC: "is_public",
    QUERY_IS_HIDDEN: "is_hidden",
}


def parse_query_location(data: bytes) -> list[dict]:
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
        AssertionError: A data-size invariant was violated (e.g. the parser's
            byte cursor doesn't land exactly on `total_size + 4` at function
            exit, or a known flag's payload is the wrong width). Detects
            wire-format drift.
        struct.error: A read would go past the buffer end (truncated input).
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
                f"malformed query buffer: field_size={field_size} < 4 at pos={pos - 2}"
            )
            flag = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            data_size = field_size - 4

            if flag == QUERY_END:
                break

            value_start = pos
            if flag in _FLAG_TO_KEY_U32:
                assert data_size == 4, f"unexpected data_size={data_size} for u32 flag 0x{flag:x}"
                card[_FLAG_TO_KEY_U32[flag]] = struct.unpack_from(
                    "<I",
                    data,
                    value_start,
                )[0]
            elif flag in _FLAG_TO_KEY_I32:
                assert data_size == 4, f"unexpected data_size={data_size} for i32 flag 0x{flag:x}"
                card[_FLAG_TO_KEY_I32[flag]] = struct.unpack_from(
                    "<i",
                    data,
                    value_start,
                )[0]
            elif flag in _FLAG_TO_KEY_U8:
                assert data_size == 1, f"unexpected data_size={data_size} for u8 flag 0x{flag:x}"
                card[_FLAG_TO_KEY_U8[flag]] = data[value_start]
            elif flag == QUERY_RACE:
                assert data_size == 8, f"unexpected data_size={data_size} for QUERY_RACE"
                card["race"] = struct.unpack_from("<Q", data, value_start)[0]
            elif flag == QUERY_LINK:
                assert data_size == 8, f"unexpected data_size={data_size} for QUERY_LINK"
                card["link_rating"] = struct.unpack_from(
                    "<I",
                    data,
                    value_start,
                )[0]
                card["link_marker"] = struct.unpack_from(
                    "<I",
                    data,
                    value_start + 4,
                )[0]
            elif flag == QUERY_OVERLAY_CARD:
                count = struct.unpack_from("<I", data, value_start)[0]
                assert data_size == 4 + 4 * count, (
                    f"QUERY_OVERLAY_CARD data_size mismatch: "
                    f"got {data_size}, expected {4 + 4 * count}"
                )
                card["overlay_cards"] = [
                    struct.unpack_from("<I", data, value_start + 4 + j * 4)[0] for j in range(count)
                ]
            elif flag == QUERY_COUNTERS:
                count = struct.unpack_from("<I", data, value_start)[0]
                assert data_size == 4 + 4 * count, (
                    f"QUERY_COUNTERS data_size mismatch: got {data_size}, expected {4 + 4 * count}"
                )
                card["counters"] = [
                    struct.unpack_from("<I", data, value_start + 4 + j * 4)[0] for j in range(count)
                ]
            else:
                raise ValueError(f"unknown query flag 0x{flag:x} at pos={value_start - 6}")
            pos += data_size

        cards.append(card)
        seq += 1

    assert pos == end, (
        f"query buffer parse drift: pos={pos} but end={end} (total_size={total_size})"
    )
    return cards
