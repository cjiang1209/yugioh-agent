"""Unit tests for the MUD protocol state machine.

Uses a FakeConnection that feeds canned server lines and records sent lines.
No MUD server required.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace

import pytest

from yugioh_mud.config import GUEST_CONFIG, HOST_CONFIG
from yugioh_mud.protocol import MUDProtocol, State


class FakeConnection:
    """Test double: feeds canned recv lines and records sent lines."""

    def __init__(self, lines: list[str]) -> None:
        self._inbox: deque[str] = deque(lines)
        self.sent: list[str] = []

    async def send_line(self, text: str) -> None:
        self.sent.append(text)

    async def recv_line(self) -> str:
        if not self._inbox:
            raise StopIteration("No more canned lines")
        return self._inbox.popleft()

    def add_lines(self, lines: list[str]) -> None:
        self._inbox.extend(lines)


def _run(coro):
    return asyncio.run(coro)


# Canned server lines for a minimal RPS-win → decision → duel sequence.
# The Menu sends each item as a separate WebSocket frame.
RPS_WIN_TO_DUEL = [
    "Select an item:",
    "[1] Rock",
    "[2] Paper",
    "[3] Scissors",
    "Type a number or @abort to abort.",
    "Here are the results:",
    "You chose Rock. Opponent chose Scissors.",
    "Since you've won, you may now choose who will go first.",
    "Select an item:",
    "[1] Yes",
    "[2] No",
    "Type a number or @abort to abort.",
    "Duel created.",
]


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_host_login(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert conn.sent[0] == "Player1"
        assert conn.sent[1] == "player1pass"

    def test_guest_login(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert conn.sent[0] == "Player2"
        assert conn.sent[1] == "player2pass"

    def test_account_not_found(self):
        lines = [
            "Nickname (or new to create a new account):",
            "That account doesn't exist. Type new to create a new account.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert conn.sent == ["Player1"]

    def test_wrong_password(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Wrong password.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert conn.sent == ["Player1", "player1pass"]

    def test_banned_account(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "You have been banned and thus may not log in.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert conn.sent == ["Player1", "player1pass"]

    def test_custom_nickname_password(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG, nickname="CustomBot", password="secret123")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert conn.sent[0] == "CustomBot"
        assert conn.sent[1] == "secret123"


# ---------------------------------------------------------------------------
# Room Setup — Host
# ---------------------------------------------------------------------------

class TestHostRoomSetup:
    def test_host_sends_room_commands_and_starts_on_guest_deck(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # Room created ack (first line triggers room commands)
            "Room created.",
            # Guest lifecycle
            "Player2 joined this room.",
            "Player2 was moved into team 2.",
            "Player2 loaded a deck.",
            # RPS
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG, deck="blue_eyes")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        # Login
        assert conn.sent[0] == "Player1"
        assert conn.sent[1] == "player1pass"
        # Lobby → create
        assert conn.sent[2] == "create"
        # Room commands (sent on first recv after create)
        assert conn.sent[3] == "banlist none"
        assert conn.sent[4] == "finish"
        assert conn.sent[5] == "move 1"
        assert conn.sent[6] == "deck blue_eyes"
        # start after guest loads deck
        assert conn.sent[7] == "start"

    def test_host_deck_not_found(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # Room ack, then deck error
            "Room created.",
            "Deck doesn't exist or isn't publically available.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG, deck="nonexistent")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert "deck nonexistent" in conn.sent


# ---------------------------------------------------------------------------
# Room Setup — Guest
# ---------------------------------------------------------------------------

class TestGuestRoomSetup:
    def test_guest_joins_and_sends_deck(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # Join ack triggers room commands
            "Joined Player1's room.",
            # Wait for host to start → eventually RPS
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG, deck="starter")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert conn.sent[0] == "Player2"
        assert conn.sent[1] == "player2pass"
        assert conn.sent[2] == "join Player1"
        assert conn.sent[3] == "move 2"
        assert conn.sent[4] == "deck starter"

    def test_guest_joins_custom_host(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            "Joined Bot99's room.",
            *RPS_WIN_TO_DUEL,
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG, join="Bot99")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert conn.sent[2] == "join Bot99"

    def test_guest_deck_not_found(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # Join ack, then deck error
            "Joined Player1's room.",
            "Deck doesn't exist or isn't publically available.",
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG, deck="nonexistent")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert "deck nonexistent" in conn.sent

    def test_guest_join_player_not_online(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            "This player isn't online.",
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG, join="Nobody")
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert conn.sent[2] == "join Nobody"

    def test_guest_join_no_room(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            "This player currently doesn't prepare to duel or you may not enter the room.",
        ]
        conn = FakeConnection(lines)
        config = replace(GUEST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.FINISHED
        assert conn.sent[2] == "join Player1"


# ---------------------------------------------------------------------------
# RPS
# ---------------------------------------------------------------------------

class TestRPS:
    def test_rps_won_then_decision(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # RPS menu
            "Select an item:",
            "[1] Rock",
            "[2] Paper",
            "[3] Scissors",
            "Type a number or @abort to abort.",
            "Here are the results:",
            "Player1 has chosen Rock.",
            "Player2 has chosen Scissors.",
            "Since you've won, you may now choose who will go first.",
            # Decision menu
            "Select an item:",
            "[1] Yes",
            "[2] No",
            "Type a number or @abort to abort.",
            "Duel created.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.DUEL
        create_idx = conn.sent.index("create")
        post_create = conn.sent[create_idx + 1:]
        # Last two sends: one RPS choice in 1-3, then decision "1"
        assert post_create[-2] in ("1", "2", "3")
        assert post_create[-1] == "1"  # decision: go first

    def test_rps_lost_wait_for_duel(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # RPS menu
            "Select an item:",
            "[1] Rock",
            "[2] Paper",
            "[3] Scissors",
            "Type a number or @abort to abort.",
            "Here are the results:",
            "Player1 has chosen Rock.",
            "Player2 has chosen Paper.",
            "You've lost, so you must wait for your opponent to choose if they want to go first.",
            "Duel created.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.DUEL
        assert proto._rps_winner is False
        # Exactly one numeric send after create: the RPS choice, no decision
        create_idx = conn.sent.index("create")
        numeric_after_create = [
            s for s in conn.sent[create_idx + 1:]
            if s in ("1", "2", "3")
        ]
        assert len(numeric_after_create) == 1

    def test_rps_tie_then_retry(self):
        lines = [
            "Nickname (or new to create a new account):",
            "Password:",
            "Welcome to the game.",
            # First RPS — tie
            "Select an item:",
            "[1] Rock",
            "[2] Paper",
            "[3] Scissors",
            "Type a number or @abort to abort.",
            "Here are the results:",
            "Player1 has chosen Rock.",
            "Player2 has chosen Rock.",
            "Noone will be the victor this round.",
            # Second RPS — win
            "Select an item:",
            "[1] Rock",
            "[2] Paper",
            "[3] Scissors",
            "Type a number or @abort to abort.",
            "Here are the results:",
            "Player1 has chosen Paper.",
            "Player2 has chosen Rock.",
            "Since you've won, you may now choose who will go first.",
            # Decision
            "Select an item:",
            "[1] Yes",
            "[2] No",
            "Type a number or @abort to abort.",
            "Duel created.",
        ]
        conn = FakeConnection(lines)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)
        _run(proto.run())

        assert proto.state == State.DUEL
        create_idx = conn.sent.index("create")
        numeric_after_create = [
            s for s in conn.sent[create_idx + 1:]
            if s in ("1", "2", "3")
        ]
        # Two RPS choices + one decision = 3 numeric sends
        assert len(numeric_after_create) == 3
        assert numeric_after_create[-1] == "1"  # decision: go first
