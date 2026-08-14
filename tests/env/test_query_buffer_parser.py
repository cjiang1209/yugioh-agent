"""Unit tests for the OCG_DuelQueryLocation buffer parser.

These tests synthesize wire bytes per the documented format and verify
that `parse_query_location` produces the expected dict structure.

Wire format (edo9300):

    [0..3]    uint32  total_data_size       (bytes after this header)
    [4..]     per-slot loop until total_data exhausted:
                empty slot:  int16(0)        — 2 bytes
                card:        repeated field blocks until QUERY_END:
                               uint16 field_size  (bytes for flag + value)
                               uint32 flag
                               bytes  value[field_size - 4]
                             terminator: uint16(4) + uint32(QUERY_END)

The helpers below build correctly-sized field blocks; tests assemble
slots and a header to produce realistic buffers.
"""

from __future__ import annotations

import struct

import pytest

from yugioh_core.constants import (
    POS_FACEUP_ATTACK,
    QUERY_CODE,
    QUERY_END,
    QUERY_IS_PUBLIC,
    QUERY_OVERLAY_CARD,
    QUERY_OWNER,
    QUERY_POSITION,
    QUERY_RACE,
    QUERY_TYPE,
    TYPE_EFFECT,
    TYPE_MONSTER,
)
from yugioh_core.query_buffer import parse_query_location

# ─── Wire-format builders ───────────────────────────────────────────────


def _u32_field(flag: int, value: int) -> bytes:
    """uint16(field_size=8) + uint32(flag) + uint32(value) → 10 bytes."""
    return struct.pack("<HII", 8, flag, value)


def _u8_field(flag: int, value: int) -> bytes:
    """uint16(field_size=5) + uint32(flag) + uint8(value) → 7 bytes."""
    return struct.pack("<HIB", 5, flag, value)


def _u64_field(flag: int, value: int) -> bytes:
    """uint16(field_size=12) + uint32(flag) + uint64(value) → 14 bytes."""
    return struct.pack("<HIQ", 12, flag, value)


def _terminator() -> bytes:
    """QUERY_END terminator block: uint16(4) + uint32(QUERY_END) → 6 bytes."""
    return struct.pack("<HI", 4, QUERY_END)


def _empty_slot() -> bytes:
    """An MZONE/SZONE empty slot: int16(0) → 2 bytes."""
    return struct.pack("<h", 0)


def _wrap(body: bytes) -> bytes:
    """Prepend the uint32 total_data_size header."""
    return struct.pack("<I", len(body)) + body


# ─── Tests ──────────────────────────────────────────────────────────────


def test_empty_buffer_returns_empty_list():
    """A buffer with total_size=0 (just the 4-byte header) parses to []."""
    data = struct.pack("<I", 0)
    assert parse_query_location(data) == []


def test_single_card_single_field():
    """One slot with one QUERY_CODE field + QUERY_END terminator."""
    body = _u32_field(QUERY_CODE, 89631139) + _terminator()
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [{"sequence": 0, "code": 89631139}]


def test_single_card_multiple_fields():
    """One slot with QUERY_CODE + QUERY_POSITION + QUERY_TYPE + QUERY_END."""
    body = (
        _u32_field(QUERY_CODE, 12345)
        + _u32_field(QUERY_POSITION, POS_FACEUP_ATTACK)
        + _u32_field(QUERY_TYPE, 0x21)  # MONSTER | EFFECT
        + _terminator()
    )
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [
        {
            "sequence": 0,
            "code": 12345,
            "position": POS_FACEUP_ATTACK,
            "type": TYPE_MONSTER | TYPE_EFFECT,
        }
    ]


def test_empty_slot_emits_empty_dict():
    """Three empty slots, then one populated card → sequence on the card is 3."""
    body = (
        _empty_slot() + _empty_slot() + _empty_slot() + _u32_field(QUERY_CODE, 999) + _terminator()
    )
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [
        {},
        {},
        {},
        {"sequence": 3, "code": 999},
    ]


def test_multiple_cards_no_empty_slots():
    """Three populated cards in a row → sequences 0/1/2."""
    body = (
        _u32_field(QUERY_CODE, 100)
        + _terminator()
        + _u32_field(QUERY_CODE, 200)
        + _terminator()
        + _u32_field(QUERY_CODE, 300)
        + _terminator()
    )
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [
        {"sequence": 0, "code": 100},
        {"sequence": 1, "code": 200},
        {"sequence": 2, "code": 300},
    ]


def test_unknown_flag_raises_valueerror():
    """An unrecognized flag must raise ValueError, not silently skip.

    Silent skip would let the model train on incomplete observations
    indefinitely if the engine submodule were updated to emit a new
    field; raising forces the parser to be updated alongside the engine.
    """
    UNKNOWN_FLAG = 0x4000000  # not in any of the _FLAG_TO_KEY_* tables
    body = _u32_field(UNKNOWN_FLAG, 42) + _terminator()
    data = _wrap(body)
    with pytest.raises(ValueError, match=r"unknown query flag 0x4000000"):
        parse_query_location(data)


def test_truncated_buffer_raises():
    """If `total_size` claims more bytes than the buffer ships, parsing
    must fail loudly. Either struct.error (mid-field read past EOF) or
    AssertionError (the final pos==end check) is acceptable — both
    indicate the same underlying issue."""
    full = _wrap(_u32_field(QUERY_CODE, 1) + _terminator())
    truncated = full[:-2]  # chop off the last 2 bytes of QUERY_END terminator
    with pytest.raises((struct.error, AssertionError)):
        parse_query_location(truncated)


def test_size_mismatch_assertion_fires():
    """If `total_size` lies about the body length (claims too few bytes),
    the parser ends with leftover bytes and the final assertion fires."""
    body = _u32_field(QUERY_CODE, 1) + _terminator()
    bogus_header = struct.pack("<I", len(body) - 2)
    data = bogus_header + body
    with pytest.raises(AssertionError, match=r"query buffer parse drift"):
        parse_query_location(data)


def test_query_overlay_card_variable_length():
    """QUERY_OVERLAY_CARD has a count + N×u32 payload. Verify the
    variable-length decoder produces the right list."""
    overlays = [11, 22, 33]
    payload = struct.pack("<I", len(overlays)) + b"".join(struct.pack("<I", v) for v in overlays)
    field = struct.pack("<HI", 4 + len(payload), QUERY_OVERLAY_CARD) + payload
    body = field + _terminator()
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [{"sequence": 0, "overlay_cards": [11, 22, 33]}]


def test_u8_field_decodes_correctly():
    """QUERY_OWNER, QUERY_IS_PUBLIC, QUERY_IS_HIDDEN are u8 fields
    (field_size=5). Verify the u8 decoder and the dict keys."""
    body = _u8_field(QUERY_OWNER, 1) + _u8_field(QUERY_IS_PUBLIC, 1) + _terminator()
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [{"sequence": 0, "owner": 1, "is_public": 1}]


def test_u64_race_field_decodes_correctly():
    """QUERY_RACE is u64 (field_size=12). Verify decode of a value
    that exceeds 32 bits."""
    race_val = (1 << 40) | 0x1234  # forces u64 rather than u32
    body = _u64_field(QUERY_RACE, race_val) + _terminator()
    data = _wrap(body)
    result = parse_query_location(data)
    assert result == [{"sequence": 0, "race": race_val}]
