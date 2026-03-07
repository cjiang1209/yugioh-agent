"""Test binary message parser."""

import struct

import pytest

from yugioh_env.message_parser import BinaryReader, parse_messages
from yugioh_env.constants import MSG_NEW_TURN, MSG_WIN, MSG_SELECT_YESNO


def test_binary_reader_u8():
    r = BinaryReader(b"\x42")
    assert r.u8() == 0x42


def test_binary_reader_u16():
    r = BinaryReader(struct.pack("<H", 1234))
    assert r.u16() == 1234


def test_binary_reader_u32():
    r = BinaryReader(struct.pack("<I", 0xDEADBEEF))
    assert r.u32() == 0xDEADBEEF


def test_binary_reader_i32():
    r = BinaryReader(struct.pack("<i", -1))
    assert r.i32() == -1


def test_binary_reader_u64():
    r = BinaryReader(struct.pack("<Q", 0x123456789ABCDEF0))
    assert r.u64() == 0x123456789ABCDEF0


def test_parse_new_turn():
    """Parse a synthetic MSG_NEW_TURN message."""
    # Build: length(4) + msg_type(1) + player(1)
    payload = bytes([MSG_NEW_TURN, 0])  # player 0
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    assert messages[0]["msg_type"] == MSG_NEW_TURN
    assert messages[0]["player"] == 0


def test_parse_win():
    """Parse a synthetic MSG_WIN message."""
    payload = bytes([MSG_WIN, 1, 0])  # player 1, reason 0
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    assert messages[0]["msg_type"] == MSG_WIN
    assert messages[0]["player"] == 1


def test_parse_multiple_messages():
    """Parse multiple concatenated messages."""
    msg1_payload = bytes([MSG_NEW_TURN, 0])
    msg2_payload = bytes([MSG_NEW_TURN, 1])
    data = (
        struct.pack("<I", len(msg1_payload)) + msg1_payload
        + struct.pack("<I", len(msg2_payload)) + msg2_payload
    )
    messages = parse_messages(data)
    assert len(messages) == 2
    assert messages[0]["player"] == 0
    assert messages[1]["player"] == 1


def test_parse_empty_buffer():
    """Empty buffer should return no messages."""
    messages = parse_messages(b"")
    assert messages == []


def test_parse_select_yesno():
    """Parse a synthetic MSG_SELECT_YESNO message."""
    payload = bytes([MSG_SELECT_YESNO, 0]) + struct.pack("<Q", 200)
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    assert messages[0]["msg_type"] == MSG_SELECT_YESNO
    assert messages[0]["player"] == 0
    assert messages[0]["desc"] == 200
