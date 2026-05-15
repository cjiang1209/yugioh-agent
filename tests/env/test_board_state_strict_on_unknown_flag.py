"""Pin the deliberate behavior change: the web UI's board-state path
now raises on unknown flags, where the old lenient parser would
silently skip (and silently desync the byte cursor).
"""

from __future__ import annotations

import struct

import pytest

from yugioh_core.constants import QUERY_CODE, QUERY_END
from yugioh_core.query_buffer import parse_query_location


def _u32_field(flag: int, value: int) -> bytes:
    """uint16(field_size=8) + uint32(flag) + uint32(value) → 10 bytes."""
    return struct.pack("<HII", 8, flag, value)


def _terminator() -> bytes:
    """QUERY_END terminator block: uint16(4) + uint32(QUERY_END) → 6 bytes."""
    return struct.pack("<HI", 4, QUERY_END)


def _wrap(body: bytes) -> bytes:
    """Prepend the uint32 total_data_size header."""
    return struct.pack("<I", len(body)) + body


def test_board_state_path_raises_on_unknown_flag():
    """Synthesize a wire buffer with an unknown flag and confirm the
    unified parser (which the web UI now consumes) raises ValueError.
    The lenient parser this replaces would have silently skipped the
    field AND desynced the byte cursor for every subsequent field."""
    UNKNOWN_FLAG = 0x10000000  # not in any _FLAG_TO_KEY_* table
    body = _u32_field(QUERY_CODE, 89631139) + _u32_field(UNKNOWN_FLAG, 42) + _terminator()
    data = _wrap(body)
    with pytest.raises(ValueError, match=r"unknown query flag 0x10000000"):
        parse_query_location(data)
