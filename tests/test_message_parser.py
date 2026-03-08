"""Test binary message parser."""

import struct

import pytest

from yugioh_env.message_parser import BinaryReader, parse_messages
from yugioh_env.constants import (
    MSG_NEW_TURN, MSG_WIN, MSG_SELECT_YESNO,
    MSG_SELECT_IDLECMD, MSG_SELECT_BATTLECMD,
)


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


def _build_idlecmd_payload(
    player=0,
    summonable=(),
    sp_summonable=(),
    repositionable=(),
    mset=(),
    sset=(),
    activatable=(),
    to_bp=0,
    to_ep=0,
    shuffle_hand=0,
):
    """Build a MSG_SELECT_IDLECMD binary payload matching the C++ engine format.

    Normal cards: code(u32) + controller(u8) + location(u8) + sequence(u32).
    Repositionable: code(u32) + controller(u8) + location(u8) + sequence(u8).
    Activatable: code(u32) + controller(u8) + location(u8) + sequence(u32) + desc(u64) + client_mode(u8).
    """
    buf = bytes([MSG_SELECT_IDLECMD, player])
    # Helper for standard card list: code(u32) + con(u8) + loc(u8) + seq(u32)
    def pack_standard(cards):
        data = struct.pack("<I", len(cards))
        for code, con, loc, seq in cards:
            data += struct.pack("<IBBI", code, con, loc, seq)
        return data
    buf += pack_standard(summonable)
    buf += pack_standard(sp_summonable)
    # Repositionable: code(u32) + con(u8) + loc(u8) + seq(u8)
    buf += struct.pack("<I", len(repositionable))
    for code, con, loc, seq in repositionable:
        buf += struct.pack("<IBBB", code, con, loc, seq)
    buf += pack_standard(mset)
    buf += pack_standard(sset)
    # Activatable: code(u32) + con(u8) + loc(u8) + seq(u32) + desc(u64) + client_mode(u8)
    buf += struct.pack("<I", len(activatable))
    for code, con, loc, seq, desc, cm in activatable:
        buf += struct.pack("<IBBIQB", code, con, loc, seq, desc, cm)
    buf += bytes([to_bp, to_ep, shuffle_hand])
    return buf


def test_parse_select_idlecmd_repositionable_sequence_u8():
    """Repositionable cards use uint8 sequence (not uint32)."""
    payload = _build_idlecmd_payload(
        repositionable=[(12345, 0, 4, 3)],
    )
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["msg_type"] == MSG_SELECT_IDLECMD
    assert len(msg["repositionable"]) == 1
    assert msg["repositionable"][0]["code"] == 12345
    assert msg["repositionable"][0]["sequence"] == 3


def test_parse_select_idlecmd_activatable_client_mode():
    """Activatable cards include a client_mode byte after desc."""
    payload = _build_idlecmd_payload(
        activatable=[(99999, 0, 2, 1, 500, 7)],
    )
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    msg = messages[0]
    assert len(msg["activatable"]) == 1
    assert msg["activatable"][0]["code"] == 99999
    assert msg["activatable"][0]["desc"] == 500
    assert msg["activatable"][0]["client_mode"] == 7


def test_parse_select_idlecmd_mixed():
    """Parse an idle cmd with multiple card categories."""
    payload = _build_idlecmd_payload(
        player=0,
        summonable=[(100, 0, 2, 0)],
        repositionable=[(200, 0, 4, 2), (300, 0, 4, 5)],
        activatable=[(400, 0, 2, 0, 1000, 1)],
        to_bp=1,
        to_ep=1,
    )
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["player"] == 0
    assert len(msg["summonable"]) == 1
    assert msg["summonable"][0]["code"] == 100
    assert len(msg["repositionable"]) == 2
    assert msg["repositionable"][0]["sequence"] == 2
    assert msg["repositionable"][1]["sequence"] == 5
    assert len(msg["activatable"]) == 1
    assert msg["activatable"][0]["client_mode"] == 1
    assert msg["to_bp"] == 1
    assert msg["to_ep"] == 1


def _build_battlecmd_payload(player=0, activatable=(), attackable=(), to_m2=0, to_ep=0):
    """Build a MSG_SELECT_BATTLECMD payload matching the C++ engine format.

    Activatable: code(u32) + con(u8) + loc(u8) + seq(u32) + desc(u64) + client_mode(u8).
    Attackable:  code(u32) + con(u8) + loc(u8) + seq(u8) + direct_attackable(u8).
    """
    buf = bytes([MSG_SELECT_BATTLECMD, player])
    # Activatable
    buf += struct.pack("<I", len(activatable))
    for code, con, loc, seq, desc, cm in activatable:
        buf += struct.pack("<IBBIQB", code, con, loc, seq, desc, cm)
    # Attackable
    buf += struct.pack("<I", len(attackable))
    for code, con, loc, seq, direct in attackable:
        buf += struct.pack("<IBBBB", code, con, loc, seq, direct)
    buf += bytes([to_m2, to_ep])
    return buf


def test_parse_select_battlecmd_client_mode():
    """Battle cmd activatable cards include client_mode byte."""
    payload = _build_battlecmd_payload(
        activatable=[(55555, 0, 4, 0, 999, 3)],
        attackable=[(77777, 0, 4, 1, 0)],
        to_m2=1,
    )
    msg_data = struct.pack("<I", len(payload)) + payload
    messages = parse_messages(msg_data)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["msg_type"] == MSG_SELECT_BATTLECMD
    assert len(msg["activatable"]) == 1
    assert msg["activatable"][0]["code"] == 55555
    assert msg["activatable"][0]["desc"] == 999
    assert msg["activatable"][0]["client_mode"] == 3
    assert len(msg["attackable"]) == 1
    assert msg["attackable"][0]["code"] == 77777
    assert msg["to_m2"] == 1
    assert msg["to_ep"] == 0
