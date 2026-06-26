"""Tests for pending chain buffer in GameState and observation."""

from __future__ import annotations

import numpy as np

from yugioh_core.constants import MSG_CHAIN_END, MSG_CHAINING
from yugioh_core.encoding import (
    CHAIN_ENTRY_FEATURES,
    MAX_PENDING_CHAIN,
    decode_u32,
    encode_chain_entry,
)
from yugioh_env.game_state import GameState


def test_encode_chain_entry_roundtrip():
    """encode_chain_entry produces correct byte layout."""
    entry = encode_chain_entry(
        code=12345,
        desc=0xABCD_0000_0000_0003,
        controller=1,
        location=0x04,
        sequence=2,
        chain_link=1,
    )
    assert entry.shape == (CHAIN_ENTRY_FEATURES,)
    assert entry.dtype == np.uint8
    assert decode_u32(entry, 0) == 12345
    assert entry[12] == 1  # controller
    assert entry[13] == 0x04  # location (mzone)
    assert entry[14] == 2  # sequence
    assert entry[15] == 1  # chain_link


def test_gamestate_appends_on_chaining():
    """MSG_CHAINING appends an entry to pending_chain."""
    gs = GameState()
    gs.update(
        {
            "msg_type": MSG_CHAINING,
            "code": 99999,
            "controller": 0,
            "location": 0x08,
            "sequence": 1,
            "desc": 0x1234_0000_0000_0005,
        }
    )
    assert gs.chain_count == 1
    assert decode_u32(gs.pending_chain[0], 0) == 99999
    assert gs.pending_chain[0, 15] == 1  # chain_link
    # Second entry is still empty
    assert gs.pending_chain[1].sum() == 0


def test_gamestate_clears_on_chain_end():
    """MSG_CHAIN_END clears the pending_chain buffer."""
    gs = GameState()
    gs.update(
        {
            "msg_type": MSG_CHAINING,
            "code": 11111,
            "controller": 0,
            "location": 0x02,
            "sequence": 0,
            "desc": 0,
        }
    )
    assert gs.chain_count == 1
    gs.update({"msg_type": MSG_CHAIN_END})
    assert gs.chain_count == 0
    assert gs.pending_chain.sum() == 0


def test_gamestate_reset_clears_chain():
    """GameState.reset() zeros the chain buffer."""
    gs = GameState()
    gs.update(
        {
            "msg_type": MSG_CHAINING,
            "code": 55555,
            "controller": 1,
            "location": 0x04,
            "sequence": 3,
            "desc": 0,
        }
    )
    gs.reset()
    assert gs.chain_count == 0
    assert gs.pending_chain.sum() == 0


def test_gamestate_multiple_chain_links():
    """Multiple MSG_CHAINING events fill sequential buffer entries."""
    gs = GameState()
    for i in range(3):
        gs.update(
            {
                "msg_type": MSG_CHAINING,
                "code": 10000 + i,
                "controller": i % 2,
                "location": 0x04,
                "sequence": i,
                "desc": 0,
            }
        )
    assert gs.chain_count == 3
    for i in range(3):
        assert decode_u32(gs.pending_chain[i], 0) == 10000 + i
        assert gs.pending_chain[i, 15] == i + 1  # chain_link 1-based
    # Remaining entries empty
    assert gs.pending_chain[3:].sum() == 0


def test_gamestate_chain_buffer_overflow():
    """Chain entries beyond MAX_PENDING_CHAIN are silently dropped."""
    gs = GameState()
    for i in range(MAX_PENDING_CHAIN + 2):
        gs.update(
            {
                "msg_type": MSG_CHAINING,
                "code": i,
                "controller": 0,
                "location": 0x04,
                "sequence": 0,
                "desc": 0,
            }
        )
    assert gs.chain_count == MAX_PENDING_CHAIN + 2
    # Buffer has exactly MAX_PENDING_CHAIN entries filled
    assert all(decode_u32(gs.pending_chain[i], 0) == i for i in range(MAX_PENDING_CHAIN))
