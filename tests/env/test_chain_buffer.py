"""Tests for the pending-chain lifecycle model in GameState + its encoding."""

from __future__ import annotations

import pytest

from yugioh_core.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_CHAIN_DISABLED,
    MSG_CHAIN_END,
    MSG_CHAIN_NEGATED,
    MSG_CHAIN_SOLVED,
    MSG_CHAIN_SOLVING,
    MSG_CHAINED,
    MSG_CHAINING,
)
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
    gs.update(
        _chaining(
            99999, controller=0, location=LOCATION_SZONE, sequence=1, desc=0x1234_0000_0000_0005
        )
    )
    assert gs.chain_count == 1
    assert len(gs.pending_chain) == 1
    link = gs.pending_chain[0]
    assert isinstance(link, ChainLink)
    assert link.code == 99999
    assert link.controller == 0  # raw engine controller, unrelativized
    assert link.location == LOCATION_SZONE
    assert link.sequence == 1
    assert link.desc == 0x1234_0000_0000_0005
    assert link.chain_link == 1
    assert link.status == ChainStatus.BUILDING


def test_chain_end_clears_list():
    gs = GameState()
    gs.update(_chaining(11111, controller=0, location=LOCATION_HAND, sequence=0))
    assert gs.chain_count == 1
    gs.update({"msg_type": MSG_CHAIN_END})
    assert gs.chain_count == 0
    assert gs.pending_chain == []


def test_reset_clears_list():
    gs = GameState()
    gs.update(_chaining(55555, controller=1, location=LOCATION_MZONE, sequence=3))
    gs.reset()
    assert gs.chain_count == 0
    assert gs.pending_chain == []


def test_multiple_links_get_sequential_chain_link_numbers():
    gs = GameState()
    for i in range(3):
        gs.update(_chaining(10000 + i, controller=i % 2, location=LOCATION_MZONE, sequence=i))
    assert gs.chain_count == 3
    assert [lnk.code for lnk in gs.pending_chain] == [10000, 10001, 10002]
    assert [lnk.chain_link for lnk in gs.pending_chain] == [1, 2, 3]


def test_list_holds_all_links_beyond_max():
    """The list is uncapped; MAX_PENDING_CHAIN applies only at encode time."""
    gs = GameState()
    for i in range(MAX_PENDING_CHAIN + 2):
        gs.update(_chaining(i, controller=0, location=LOCATION_MZONE, sequence=0))
    assert gs.chain_count == MAX_PENDING_CHAIN + 2
    assert len(gs.pending_chain) == MAX_PENDING_CHAIN + 2


def _build(gs, n):
    for i in range(n):
        gs.update(_chaining(1000 + i, controller=0, location=LOCATION_MZONE, sequence=i))


def test_solving_then_solved_sets_status():
    gs = GameState()
    _build(gs, 2)
    gs.update({"msg_type": MSG_CHAIN_SOLVING, "chain_link": 2})
    assert gs.pending_chain[1].status == ChainStatus.SOLVING
    gs.update({"msg_type": MSG_CHAIN_SOLVED, "chain_link": 2})
    assert gs.pending_chain[1].status == ChainStatus.SOLVED
    # Link 1 untouched, still building
    assert gs.pending_chain[0].status == ChainStatus.BUILDING


@pytest.mark.parametrize(
    "terminal_msg, terminal_status",
    [
        (MSG_CHAIN_NEGATED, ChainStatus.NEGATED),
        (MSG_CHAIN_DISABLED, ChainStatus.DISABLED),
    ],
)
def test_terminal_status_takes_precedence_over_solved(terminal_msg, terminal_status):
    gs = GameState()
    _build(gs, 1)
    gs.update({"msg_type": terminal_msg, "chain_link": 1})
    assert gs.pending_chain[0].status == terminal_status
    # A later SOLVED for the same link must NOT overwrite the terminal status.
    gs.update({"msg_type": MSG_CHAIN_SOLVED, "chain_link": 1})
    assert gs.pending_chain[0].status == terminal_status


def test_chained_message_does_not_change_status():
    gs = GameState()
    _build(gs, 1)
    gs.update({"msg_type": MSG_CHAINED, "chain_link": 1})
    assert gs.pending_chain[0].status == ChainStatus.BUILDING


def test_unknown_chain_link_is_ignored_not_crash():
    gs = GameState()
    _build(gs, 1)
    # No link #5 exists — must NOT raise, and must leave state intact.
    gs.update({"msg_type": MSG_CHAIN_SOLVED, "chain_link": 5})
    assert gs.pending_chain[0].status == ChainStatus.BUILDING
