"""Tests for the pending-chain lifecycle model in GameState + its encoding."""

from __future__ import annotations

from yugioh_core.constants import MSG_CHAIN_END, MSG_CHAINING
from yugioh_core.encoding import MAX_PENDING_CHAIN
from yugioh_env.game_state import ChainLink, ChainStatus, GameState


def _chaining(code, controller, location, sequence, desc=0):
    return {
        "msg_type": MSG_CHAINING,
        "code": code,
        "controller": controller,
        "location": location,
        "sequence": sequence,
        "desc": desc,
    }


def test_chaining_appends_chainlink():
    gs = GameState()
    gs.update(_chaining(99999, controller=0, location=0x08, sequence=1, desc=0x1234_0000_0000_0005))
    assert gs.chain_count == 1
    assert len(gs.pending_chain) == 1
    link = gs.pending_chain[0]
    assert isinstance(link, ChainLink)
    assert link.code == 99999
    assert link.controller == 0  # raw engine controller, unrelativized
    assert link.location == 0x08
    assert link.sequence == 1
    assert link.desc == 0x1234_0000_0000_0005
    assert link.chain_link == 1
    assert link.status == ChainStatus.BUILDING


def test_chain_end_clears_list():
    gs = GameState()
    gs.update(_chaining(11111, controller=0, location=0x02, sequence=0))
    assert gs.chain_count == 1
    gs.update({"msg_type": MSG_CHAIN_END})
    assert gs.chain_count == 0
    assert gs.pending_chain == []


def test_reset_clears_list():
    gs = GameState()
    gs.update(_chaining(55555, controller=1, location=0x04, sequence=3))
    gs.reset()
    assert gs.chain_count == 0
    assert gs.pending_chain == []


def test_multiple_links_get_sequential_chain_link_numbers():
    gs = GameState()
    for i in range(3):
        gs.update(_chaining(10000 + i, controller=i % 2, location=0x04, sequence=i))
    assert gs.chain_count == 3
    assert [lnk.code for lnk in gs.pending_chain] == [10000, 10001, 10002]
    assert [lnk.chain_link for lnk in gs.pending_chain] == [1, 2, 3]


def test_list_holds_all_links_beyond_max():
    """The list is uncapped; MAX_PENDING_CHAIN applies only at encode time."""
    gs = GameState()
    for i in range(MAX_PENDING_CHAIN + 2):
        gs.update(_chaining(i, controller=0, location=0x04, sequence=0))
    assert gs.chain_count == MAX_PENDING_CHAIN + 2
    assert len(gs.pending_chain) == MAX_PENDING_CHAIN + 2
