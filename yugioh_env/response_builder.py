"""Build binary response buffers for ygopro-core MSG_SELECT_* messages."""

from __future__ import annotations

import struct

from yugioh_core.constants import *  # noqa: F401,F403


def build_select_idlecmd_response(action_type: int, index: int) -> bytes:
    """Build response for MSG_SELECT_IDLECMD.

    action_type categories:
        0 = summon, 1 = sp_summon, 2 = reposition, 3 = mset, 4 = sset,
        5 = activate, 6 = to_bp, 7 = to_ep, 8 = shuffle
    """
    return struct.pack("<I", (index << 16) | action_type)


def build_select_battlecmd_response(action_type: int, index: int) -> bytes:
    """Build response for MSG_SELECT_BATTLECMD.

    action_type categories:
        0 = activate, 1 = attack, 2 = to_m2, 3 = to_ep
    """
    return struct.pack("<I", (index << 16) | action_type)


def build_select_card_response(indices: list[int]) -> bytes:
    """Build response for MSG_SELECT_CARD / MSG_SELECT_TRIBUTE.

    Response format (edo9300): type(int32) + size(uint32) + indices...
    type=0 uses uint32 indices at positions [2..], type=1 uses uint16, type=2 uses uint8.
    We use type=0 (uint32 indices).
    """
    return struct.pack("<iI", 0, len(indices)) + b"".join(struct.pack("<I", i) for i in indices)


def build_select_chain_response(index: int) -> bytes:
    """Build response for MSG_SELECT_CHAIN. -1 = no chain."""
    return struct.pack("<i", index)


def build_select_yesno_response(yes: bool) -> bytes:
    """Build response for MSG_SELECT_YESNO / MSG_SELECT_EFFECTYN."""
    return struct.pack("<I", 1 if yes else 0)


def build_select_option_response(index: int) -> bytes:
    """Build response for MSG_SELECT_OPTION."""
    return struct.pack("<I", index)


def build_select_position_response(position: int) -> bytes:
    """Build response for MSG_SELECT_POSITION."""
    return struct.pack("<I", position)


def build_select_place_response(player: int, location: int, sequence: int) -> bytes:
    """Build response for MSG_SELECT_PLACE / MSG_SELECT_DISFIELD."""
    return struct.pack("<BBB", player, location, sequence)


def build_sort_card_response(order: list[int]) -> bytes:
    """Build response for MSG_SORT_CARD / MSG_SORT_CHAIN."""
    return bytes(order)


def build_select_sum_response(indices: list[int]) -> bytes:
    """Build response for MSG_SELECT_SUM.

    Uses the same type-discriminated format as MSG_SELECT_CARD.
    """
    return struct.pack("<iI", 0, len(indices)) + b"".join(struct.pack("<I", i) for i in indices)


def build_select_unselect_card_response(index: int) -> bytes:
    """Build response for MSG_SELECT_UNSELECT_CARD.

    The engine expects returns[0] = 1 (exactly one selection) and returns[1] = card index.
    Or returns[0] = -1 to cancel/finish.
    """
    if index == -1:
        return struct.pack("<i", -1)
    return struct.pack("<iI", 1, index)


def build_announce_race_response(race_mask: int) -> bytes:
    """Build response for MSG_ANNOUNCE_RACE."""
    return struct.pack("<Q", race_mask)


def build_announce_attrib_response(attrib_mask: int) -> bytes:
    """Build response for MSG_ANNOUNCE_ATTRIB."""
    return struct.pack("<I", attrib_mask)


def build_announce_card_response(code: int) -> bytes:
    """Build response for MSG_ANNOUNCE_CARD."""
    return struct.pack("<I", code)


def build_announce_number_response(index: int) -> bytes:
    """Build response for MSG_ANNOUNCE_NUMBER.

    The engine reads `returns.at<int32_t>(0)` as an INDEX into the announced
    options list, NOT the announced value itself
    (third_party/ygopro-core/playerop.cpp:1109). Passing the value out-of-range
    triggers MSG_RETRY and silently forfeits the duel.
    """
    return struct.pack("<i", index)


def build_rock_paper_scissors_response(choice: int) -> bytes:
    """Build response for MSG_ROCK_PAPER_SCISSORS. 1=rock, 2=paper, 3=scissors."""
    return struct.pack("<I", choice)


def build_select_counter_response(counters: list[int]) -> bytes:
    """Build response for MSG_SELECT_COUNTER. List of counter counts per card.

    The engine reads returns.at<int16_t>(i) directly for each card — no length prefix.
    """
    return b"".join(struct.pack("<H", c) for c in counters)
