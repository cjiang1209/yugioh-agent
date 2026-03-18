"""Unit tests for the MUD protocol state machine.

Uses a FakeConnection that feeds canned server lines and records sent lines.
No MUD server required.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace

import pytest

from yugioh_mud.action_translator import ActionTranslator
from yugioh_mud.agent import PassiveAgent, RandomAgent
from yugioh_mud.config import GUEST_CONFIG, HOST_CONFIG
from yugioh_mud.protocol import MUDProtocol, State
from yugioh_mud.text_parser import MUDTextParser


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


# ---------------------------------------------------------------------------
# Duel — passive play
# ---------------------------------------------------------------------------

# Minimal login → duel-created → duel prompts → duel end sequence
LOGIN_TO_DUEL = [
    "Nickname (or new to create a new account):",
    "Password:",
    "Welcome to the game.",
    *RPS_WIN_TO_DUEL,
]


def _make_duel_proto(extra_lines: list[str]) -> tuple[FakeConnection, MUDProtocol]:
    """Create a protocol wired with duel components and canned lines."""
    conn = FakeConnection([*LOGIN_TO_DUEL, *extra_lines])
    config = replace(HOST_CONFIG)
    parser = MUDTextParser()
    agent = PassiveAgent()
    translator = ActionTranslator()
    proto = MUDProtocol(
        conn, config,
        text_parser=parser, agent=agent, action_translator=translator)
    return conn, proto


class TestDuelPassive:
    def test_duel_end_on_win(self):
        """Bot reaches FINISHED after a duel-end line."""
        conn, proto = _make_duel_proto([
            "Your turn.",
            "entering main1 phase.",
            "Select a card on which to perform an action.",
            "e: End phase.",
            "Select a card:",
            # Server response to "?" (no usable cards)
            "Select a card on which to perform an action.",
            "e: End phase.",
            "Select a card:",
            "entering end phase.",
            "You won (ran out of cards to draw).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED
        # Protocol sends "?" then agent sends "e" to end phase
        duel_sends = conn.sent[conn.sent.index("1"):]
        assert "?" in duel_sends
        assert "e" in duel_sends

    def test_duel_end_on_loss(self):
        conn, proto = _make_duel_proto([
            "You lost (LP became 0).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED

    def test_passive_declines_effects(self):
        """PassiveAgent sends 'n' for effect prompts."""
        conn, proto = _make_duel_proto([
            "Do you want to use the effect from Mirror Force in s1?",
            "entering end phase.",
            "You won (ran out of cards to draw).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED
        # After login/room/RPS commands, the duel sends should include "n"
        duel_sends = conn.sent[conn.sent.index("1"):]  # after decision "1"
        assert "n" in duel_sends

    def test_passive_cancels_chain(self):
        """PassiveAgent sends 'c' for optional chains."""
        conn, proto = _make_duel_proto([
            "Select chain (c to cancel):",
            "s1: Mirror Force",
            "Select card to chain (c = cancel):",
            "You won (ran out of cards to draw).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED
        duel_sends = conn.sent[conn.sent.index("1"):]
        assert "c" in duel_sends

    def test_passive_ends_battle(self):
        """PassiveAgent sends 'e' for battle menu."""
        conn, proto = _make_duel_proto([
            "Battle menu:",
            "a: Attack.",
            "e: End phase.",
            "Select an option:",
            # BattleCmdHandler probes "a" → attack card list → back
            "m1: Blue-Eyes White Dragon",
            "z: back.",
            "Select a card:",
            # Re-sent battle menu after "z" (discarded by handler)
            "Battle menu:",
            "a: Attack.",
            "e: End phase.",
            "Select an option:",
            # PassiveAgent picks END_PHASE → handler sends "e"
            "You lost (LP became 0).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED
        duel_sends = conn.sent[conn.sent.index("1"):]
        assert "e" in duel_sends

    def test_backward_compat_no_duel_components(self):
        """Without duel components, protocol stops at DUEL (phase 1.1)."""
        conn = FakeConnection(LOGIN_TO_DUEL)
        config = replace(HOST_CONFIG)
        proto = MUDProtocol(conn, config)  # no duel components
        _run(proto.run())
        assert proto.state == State.DUEL


# ---------------------------------------------------------------------------
# Duel — idle enrichment via "?" command
# ---------------------------------------------------------------------------

def _make_random_duel_proto(
    extra_lines: list[str],
    seed: int = 0,
) -> tuple[FakeConnection, MUDProtocol]:
    """Create protocol with RandomAgent for active play tests."""
    conn = FakeConnection([*LOGIN_TO_DUEL, *extra_lines])
    config = replace(HOST_CONFIG)
    parser = MUDTextParser()
    agent = RandomAgent(seed=seed)
    translator = ActionTranslator()
    proto = MUDProtocol(
        conn, config,
        text_parser=parser, agent=agent, action_translator=translator)
    return conn, proto


class TestIdleEnrichment:
    def test_idle_cmd_sends_question_mark(self):
        """Protocol sends '?' when receiving IDLE_CMD to get usable cards."""
        conn, proto = _make_duel_proto([
            "Your turn.",
            "entering main1 phase.",
            # IDLE_CMD prompt
            "Select a card on which to perform an action.",
            "b: Enter the battle phase.",
            "e: End phase.",
            "Select a card:",
            # Server response to "?"
            "Summonable in attack position: h1, h3",
            "Activatable: h2",
            # Re-sent idle prompt after "?"
            "Select a card on which to perform an action.",
            "b: Enter the battle phase.",
            "e: End phase.",
            "Select a card:",
            # End duel
            "You won (ran out of cards to draw).",
        ])
        _run(proto.run())
        assert proto.state == State.FINISHED
        duel_sends = conn.sent[conn.sent.index("1"):]  # after decision "1"
        assert "?" in duel_sends

    def test_idle_enrichment_cardspecs_available(self):
        """RandomAgent can pick cardspecs from enriched idle prompt."""
        # Use a seed that will pick an index action (not END_PHASE)
        # We try many seeds to find one that picks a cardspec
        for seed in range(20):
            conn, proto = _make_random_duel_proto([
                "Your turn.",
                "entering main1 phase.",
                # IDLE_CMD prompt
                "Select a card on which to perform an action.",
                "b: Enter the battle phase.",
                "e: End phase.",
                "Select a card:",
                # Server response to "?"
                "Summonable in attack position: h1, h3",
                "Settable: h2, h4",
                # Re-sent idle prompt after "?"
                "Select a card on which to perform an action.",
                "b: Enter the battle phase.",
                "e: End phase.",
                "Select a card:",
                # Idle submenu after card selection
                "s: Summon.",
                "t: Set.",
                "z: back.",
                "Select action for Dark Magician",
                # End duel
                "You won (ran out of cards to draw).",
            ], seed=seed)
            _run(proto.run())
            duel_sends = conn.sent[conn.sent.index("1"):]
            # Check if any cardspec was sent
            cardspecs_sent = [s for s in duel_sends
                              if s in ("h1", "h2", "h3", "h4")]
            if cardspecs_sent:
                return  # success — at least one seed picked a cardspec
        pytest.fail("No seed out of 20 picked a cardspec")
