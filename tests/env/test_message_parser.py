"""Test binary message parser."""

import struct

import pytest

from yugioh_core.constants import (
    LOCATION_DECK,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_ATTACK,
    MSG_BATTLE,
    MSG_BECOME_TARGET,
    MSG_CANCEL_TARGET,
    MSG_CARD_HINT,
    MSG_CARD_SELECTED,
    MSG_CARD_TARGET,
    MSG_CHAIN_DISABLED,
    MSG_CHAIN_NEGATED,
    MSG_CHAIN_SOLVED,
    MSG_CHAIN_SOLVING,
    MSG_CHAINED,
    MSG_EQUIP,
    MSG_NEW_TURN,
    MSG_RANDOM_SELECTED,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_SUM,
    MSG_SELECT_YESNO,
    MSG_SORT_CHAIN,
    MSG_WIN,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_env.message_parser import BinaryReader, parse_messages


def _pack_loc_info(controller, location, sequence, position):
    """Pack a loc_info struct: u8(con) + u8(loc) + u32(seq) + u32(pos)."""
    return struct.pack("<BBII", controller, location, sequence, position)


def _wrap_message(msg_type, body):
    """Wrap a message body with framing: u32(length) + u8(msg_type) + body."""
    payload = bytes([msg_type]) + body
    return struct.pack("<I", len(payload)) + payload


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
        struct.pack("<I", len(msg1_payload))
        + msg1_payload
        + struct.pack("<I", len(msg2_payload))
        + msg2_payload
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


# --- MSG_SELECT_SUM: loc_info includes position field ---


def test_parse_select_sum_loc_info():
    """MSG_SELECT_SUM card entries use full loc_info (with position)."""
    body = bytes([0])  # player
    body += bytes([0])  # select_type
    body += struct.pack("<I", 1000)  # target_sum
    body += struct.pack("<I", 1)  # min
    body += struct.pack("<I", 2)  # max
    # 1 must card: code + loc_info(con, loc, seq, pos) + param
    body += struct.pack("<I", 1)  # must_count
    body += struct.pack("<I", 89631139)  # code
    body += _pack_loc_info(0, LOCATION_HAND, 3, POS_FACEUP_ATTACK)
    body += struct.pack("<I", 500)  # param
    # 1 optional card
    body += struct.pack("<I", 1)  # opt_count
    body += struct.pack("<I", 46986414)
    body += _pack_loc_info(1, LOCATION_MZONE, 2, POS_FACEUP_DEFENSE)
    body += struct.pack("<I", 600)

    messages = parse_messages(_wrap_message(MSG_SELECT_SUM, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["msg_type"] == MSG_SELECT_SUM
    # Must card
    mc = msg["must_cards"][0]
    assert mc["code"] == 89631139
    assert mc["controller"] == 0
    assert mc["location"] == LOCATION_HAND
    assert mc["sequence"] == 3
    assert mc["position"] == POS_FACEUP_ATTACK
    assert mc["param"] == 500
    # Optional card
    oc = msg["optional_cards"][0]
    assert oc["code"] == 46986414
    assert oc["controller"] == 1
    assert oc["location"] == LOCATION_MZONE
    assert oc["sequence"] == 2
    assert oc["position"] == POS_FACEUP_DEFENSE
    assert oc["param"] == 600


# --- MSG_SORT_CHAIN: location is u32, not u8 ---


def test_parse_sort_chain_location_u32():
    """MSG_SORT_CHAIN uses u32 for location (same as MSG_SORT_CARD)."""
    body = bytes([0])  # player
    body += struct.pack("<I", 2)  # count
    # Card 0: code(u32) + controller(u8) + location(u32) + sequence(u32)
    body += struct.pack("<I", 100)
    body += bytes([0])
    body += struct.pack("<I", LOCATION_MZONE)
    body += struct.pack("<I", 1)
    # Card 1
    body += struct.pack("<I", 200)
    body += bytes([1])
    body += struct.pack("<I", LOCATION_SZONE)
    body += struct.pack("<I", 3)

    messages = parse_messages(_wrap_message(MSG_SORT_CHAIN, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["msg_type"] == MSG_SORT_CHAIN
    assert len(msg["cards"]) == 2
    assert msg["cards"][0]["code"] == 100
    assert msg["cards"][0]["location"] == LOCATION_MZONE
    assert msg["cards"][0]["sequence"] == 1
    assert msg["cards"][1]["code"] == 200
    assert msg["cards"][1]["controller"] == 1
    assert msg["cards"][1]["location"] == LOCATION_SZONE
    assert msg["cards"][1]["sequence"] == 3


# --- MSG_ATTACK: uses full loc_info (10 bytes) per card ---


def test_parse_attack_loc_info():
    """MSG_ATTACK uses loc_info (u8,u8,u32,u32) for attacker and target."""
    body = _pack_loc_info(0, LOCATION_MZONE, 2, POS_FACEUP_ATTACK)  # attacker
    body += _pack_loc_info(1, LOCATION_MZONE, 3, POS_FACEUP_DEFENSE)  # target

    messages = parse_messages(_wrap_message(MSG_ATTACK, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["attacker_controller"] == 0
    assert msg["attacker_location"] == LOCATION_MZONE
    assert msg["attacker_sequence"] == 2
    assert msg["target_controller"] == 1
    assert msg["target_location"] == LOCATION_MZONE
    assert msg["target_sequence"] == 3


# --- MSG_BATTLE: uses full loc_info ---


def test_parse_battle_loc_info():
    """MSG_BATTLE uses loc_info for attacker and target locations."""
    body = _pack_loc_info(0, LOCATION_MZONE, 1, POS_FACEUP_ATTACK)  # attacker loc
    body += struct.pack("<I", 2500)  # attacker_atk
    body += struct.pack("<I", 2000)  # attacker_def
    body += bytes([0])  # destroyed flag
    body += _pack_loc_info(1, LOCATION_MZONE, 0, POS_FACEUP_DEFENSE)  # target loc
    body += struct.pack("<I", 1800)  # target_atk
    body += struct.pack("<I", 1500)  # target_def
    body += bytes([1])  # destroyed flag

    messages = parse_messages(_wrap_message(MSG_BATTLE, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["attacker_controller"] == 0
    assert msg["attacker_sequence"] == 1
    assert msg["attacker_atk"] == 2500
    assert msg["attacker_def"] == 2000
    assert msg["target_controller"] == 1
    assert msg["target_sequence"] == 0
    assert msg["target_atk"] == 1800
    assert msg["target_def"] == 1500


# --- MSG_CARD_HINT: uses full loc_info ---


def test_parse_card_hint_loc_info():
    """MSG_CARD_HINT uses loc_info (10 bytes), then u8 hint_type + u64 value."""
    body = _pack_loc_info(0, LOCATION_MZONE, 2, POS_FACEUP_ATTACK)  # loc_info
    body += bytes([5])  # hint_type
    body += struct.pack("<Q", 12345678)  # value

    messages = parse_messages(_wrap_message(MSG_CARD_HINT, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["controller"] == 0
    assert msg["location"] == LOCATION_MZONE
    assert msg["sequence"] == 2
    assert msg["hint_type"] == 5
    assert msg["value"] == 12345678


# --- MSG_EQUIP: uses full loc_info × 2 ---


def test_parse_equip_loc_info():
    """MSG_EQUIP uses loc_info for equip card and target."""
    body = _pack_loc_info(0, LOCATION_SZONE, 1, POS_FACEUP_ATTACK)  # equip card
    body += _pack_loc_info(0, LOCATION_MZONE, 0, POS_FACEUP_ATTACK)  # target

    messages = parse_messages(_wrap_message(MSG_EQUIP, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["equip_controller"] == 0
    assert msg["equip_location"] == LOCATION_SZONE
    assert msg["equip_sequence"] == 1
    assert msg["target_controller"] == 0
    assert msg["target_location"] == LOCATION_MZONE
    assert msg["target_sequence"] == 0


# --- MSG_CARD_TARGET / MSG_CANCEL_TARGET: use full loc_info × 2 ---


def test_parse_card_target_loc_info():
    """MSG_CARD_TARGET uses loc_info for source and target."""
    body = _pack_loc_info(0, LOCATION_SZONE, 2, POS_FACEUP_ATTACK)
    body += _pack_loc_info(1, LOCATION_MZONE, 3, POS_FACEUP_DEFENSE)

    messages = parse_messages(_wrap_message(MSG_CARD_TARGET, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["equip_controller"] == 0
    assert msg["equip_sequence"] == 2
    assert msg["target_controller"] == 1
    assert msg["target_sequence"] == 3


def test_parse_cancel_target_loc_info():
    """MSG_CANCEL_TARGET uses loc_info for source and target."""
    body = _pack_loc_info(1, LOCATION_MZONE, 0, POS_FACEUP_ATTACK)
    body += _pack_loc_info(0, LOCATION_SZONE, 4, POS_FACEDOWN_DEFENSE)

    messages = parse_messages(_wrap_message(MSG_CANCEL_TARGET, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["equip_controller"] == 1
    assert msg["target_controller"] == 0
    assert msg["target_sequence"] == 4


# --- MSG_BECOME_TARGET: uses full loc_info per card ---


def test_parse_become_target_loc_info():
    """MSG_BECOME_TARGET uses full loc_info (with position) per card."""
    body = struct.pack("<I", 2)  # count
    body += _pack_loc_info(0, LOCATION_MZONE, 1, POS_FACEUP_ATTACK)
    body += _pack_loc_info(1, LOCATION_MZONE, 3, POS_FACEUP_DEFENSE)

    messages = parse_messages(_wrap_message(MSG_BECOME_TARGET, body))
    assert len(messages) == 1
    msg = messages[0]
    assert len(msg["cards"]) == 2
    assert msg["cards"][0]["controller"] == 0
    assert msg["cards"][0]["sequence"] == 1
    assert msg["cards"][0]["position"] == POS_FACEUP_ATTACK
    assert msg["cards"][1]["controller"] == 1
    assert msg["cards"][1]["sequence"] == 3


# --- MSG_RANDOM_SELECTED: has u8(player) prefix + loc_info per card ---


def test_parse_random_selected_player_and_loc_info():
    """MSG_RANDOM_SELECTED has a u8 player prefix then loc_info per card."""
    body = bytes([1])  # player
    body += struct.pack("<I", 1)  # count
    body += _pack_loc_info(0, LOCATION_DECK, 5, 0)

    messages = parse_messages(_wrap_message(MSG_RANDOM_SELECTED, body))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["msg_type"] == MSG_RANDOM_SELECTED
    assert msg["player"] == 1
    assert len(msg["cards"]) == 1
    assert msg["cards"][0]["controller"] == 0
    assert msg["cards"][0]["location"] == LOCATION_DECK
    assert msg["cards"][0]["sequence"] == 5


# --- MSG_CARD_SELECTED: uses full loc_info per card ---


def test_parse_card_selected_loc_info():
    """MSG_CARD_SELECTED uses full loc_info (position field present)."""
    body = struct.pack("<I", 1)  # count
    body += _pack_loc_info(0, LOCATION_MZONE, 2, POS_FACEUP_ATTACK)

    messages = parse_messages(_wrap_message(MSG_CARD_SELECTED, body))
    assert len(messages) == 1
    msg = messages[0]
    assert len(msg["cards"]) == 1
    assert msg["cards"][0]["controller"] == 0
    assert msg["cards"][0]["sequence"] == 2
    assert msg["cards"][0]["position"] == POS_FACEUP_ATTACK


def test_parser_emits_absolute_player_ids():
    """Wire-format invariant: parsers pass through u8 player IDs verbatim
    from the engine. 0 = first player, 1 = second player. NEVER relativized
    by the parser. Relativization happens downstream in the env/observation
    layer.

    This invariant is load-bearing: every relativized `controller` field in
    the observation depends on the parser side staying engine-absolute.
    """
    from yugioh_env.message_parser import _parse_select_chain, _parse_select_yesno

    # SELECT_YESNO with player=1 + a known desc.
    # Wire format: u8 player + u64 desc.
    yesno_bytes = struct.pack("<BQ", 1, 0xDEADBEEF)
    parsed = _parse_select_yesno(BinaryReader(yesno_bytes))
    assert parsed["player"] == 1, "parser must pass through engine-absolute player ID"

    # SELECT_CHAIN with two chain entries on different engine sides.
    # Wire format: u8 player + u8 spe_count + u8 forced + u32 hint_timing
    #   + u32 other_timing + u32 count + entries.
    # Each entry: u32 code + u8 controller + u8 location + u32 sequence
    #   + u32 position + u64 desc + u8 client_mode.
    chain_bytes = struct.pack(
        "<BBBIII",
        1,  # player (engine player 1 is being asked)
        0,  # spe_count
        1,  # forced
        0,  # hint_timing
        0,  # other_timing
        2,  # count: 2 chain entries
    )
    # Entry 0: card on engine player 0's side
    chain_bytes += struct.pack(
        "<IBBIIQB",
        100,  # code
        0,  # controller (engine-absolute player 0)
        LOCATION_GRAVE,
        0,
        0,
        0,
        0,  # sequence, position, desc, client_mode
    )
    # Entry 1: card on engine player 1's side
    chain_bytes += struct.pack(
        "<IBBIIQB",
        200,
        1,  # controller (engine-absolute player 1)
        0x10,
        0,
        0,
        0,
        0,
    )
    parsed = _parse_select_chain(BinaryReader(chain_bytes))
    assert parsed["player"] == 1
    assert parsed["chains"][0]["controller"] == 0, "chain entry 0 must keep absolute controller"
    assert parsed["chains"][1]["controller"] == 1, "chain entry 1 must keep absolute controller"


@pytest.mark.parametrize(
    "msg_type",
    [
        MSG_CHAINED,
        MSG_CHAIN_SOLVING,
        MSG_CHAIN_SOLVED,
        MSG_CHAIN_NEGATED,
        MSG_CHAIN_DISABLED,
    ],
)
def test_parse_chain_link_reads_single_byte(msg_type):
    buf = _wrap_message(msg_type, bytes([3]))  # chain_link = 3
    msgs = parse_messages(buf)
    assert len(msgs) == 1
    assert msgs[0]["msg_type"] == msg_type
    assert msgs[0]["chain_link"] == 3
