"""MUD protocol state machine: login → lobby → room → duel → finished."""

from __future__ import annotations

import logging
import random
from enum import Enum, auto
from typing import Protocol

from yugioh_mud.action_translator import ActionTranslator
from yugioh_mud.agent import Agent
from yugioh_mud.cmd_handler import BattleCmdHandler, IdleCmdHandler
from yugioh_mud.config import MUDBotConfig
from yugioh_mud.game_state import MUDGameState
from yugioh_mud.text_parser import (
    EventType, MUDTextParser, ParsedEvent, ParsedPrompt, PromptType,
    is_duel_end,
)

# Known DuelReader/DuelMenu re-prompt lines.  The MUD server re-sends these
# after informational commands (h, tab, tab2, score).  Used in _send_resync
# to detect end of command response.
_DUELREADER_REPROMPTS = frozenset({
    "Select a card:",       # idle DuelReader
    "Select an option:",    # battle DuelMenu
    "Enter a line of text.",  # various DuelReader prompts
})

logger = logging.getLogger(__name__)


class Connection(Protocol):
    """Minimal interface for sending/receiving lines."""

    async def send_line(self, text: str) -> None: ...
    async def recv_line(self) -> str: ...


class State(Enum):
    LOGIN_NICKNAME = auto()
    LOGIN_PASSWORD = auto()
    LOBBY = auto()
    ROOM_SETUP = auto()
    PRE_DUEL_RPS = auto()
    PRE_DUEL_DECISION = auto()
    DUEL = auto()
    FINISHED = auto()


class MUDProtocol:
    """Drives the bot through login → lobby → room → duel → finished."""

    def __init__(
        self,
        conn: Connection,
        config: MUDBotConfig,
        *,
        text_parser: MUDTextParser | None = None,
        agent: Agent | None = None,
        action_translator: ActionTranslator | None = None,
        game_state: MUDGameState | None = None,
    ) -> None:
        self.conn = conn
        self.config = config
        self.state = State.LOGIN_NICKNAME
        self._rng = random.Random(config.seed)
        self._room_commands_sent = False
        self._guest_ready = False
        self._rps_winner: bool | None = None  # True=won, False=lost, None=unknown
        # Duel-play components (optional — without these, stops at DUEL)
        self._text_parser = text_parser
        self._agent = agent
        self._action_translator = action_translator
        self._game_state = game_state
        self._resync_pending = False

    async def run(self) -> None:
        """Main loop: recv lines, dispatch by state until FINISHED.

        If no duel-play components are configured, stops at DUEL state
        (backward-compatible with phase 1.1 tests).
        """
        while self.state != State.FINISHED:
            # Stop at DUEL if no duel-play components configured
            if self.state == State.DUEL and self._text_parser is None:
                break
            line = await self.conn.recv_line()
            if self.config.verbose:
                logger.info("[%s] recv: %s", self.state.name, line)
            await self._dispatch(line)

    async def _dispatch(self, line: str) -> None:
        if self.state == State.LOGIN_NICKNAME:
            await self._handle_login_nickname(line)
        elif self.state == State.LOGIN_PASSWORD:
            await self._handle_login_password(line)
        elif self.state == State.LOBBY:
            await self._handle_lobby(line)
        elif self.state == State.ROOM_SETUP:
            await self._handle_room_setup(line)
        elif self.state == State.PRE_DUEL_RPS:
            await self._handle_rps(line)
        elif self.state == State.PRE_DUEL_DECISION:
            await self._handle_decision(line)
        elif self.state == State.DUEL:
            await self._handle_duel(line)

    # -- Login --

    async def _handle_login_nickname(self, line: str) -> None:
        if "Nickname" in line:
            await self._send(self.config.nickname)
            self.state = State.LOGIN_PASSWORD

    async def _handle_login_password(self, line: str) -> None:
        if "Password:" in line:
            await self._send(self.config.password)
            self.state = State.LOBBY
        elif "That account doesn't exist." in line:
            logger.error("Login failed: %s", line)
            self.state = State.FINISHED

    # -- Lobby --

    _LOGIN_FAILURES = (
        "Wrong password.",
        "You have been banned and thus may not log in.",
    )

    async def _handle_lobby(self, line: str) -> None:
        # Server sends one message after password: MOTD on success,
        # or an error ("Wrong password.", "...may not log in.") before disconnect.
        for pattern in self._LOGIN_FAILURES:
            if pattern in line:
                logger.error("Login failed: %s", line)
                self.state = State.FINISHED
                return
        # Any non-failure response means login succeeded (MOTD, "Reconnecting...", etc.)
        if self.config.profile == "host":
            await self._send("create")
        else:
            host = self.config.join or "Player1"
            await self._send(f"join {host}")
        self.state = State.ROOM_SETUP

    # -- Room setup --

    _ROOM_SETUP_FAILURES = (
        "Deck doesn't exist or isn't publically available.",
        "This player isn't online.",
        "This player currently doesn't prepare to duel or you may not enter the room.",
    )

    async def _handle_room_setup(self, line: str) -> None:
        for pattern in self._ROOM_SETUP_FAILURES:
            if pattern in line:
                logger.error("Room setup failed: %s", line)
                self.state = State.FINISHED
                return
        if self.config.profile == "host":
            await self._handle_room_host(line)
        else:
            await self._handle_room_guest(line)

    async def _handle_room_host(self, line: str) -> None:
        # Send initial room commands once (after any ack from create)
        if not self._room_commands_sent:
            await self._send("banlist none")
            await self._send("finish")
            await self._send("move 1")
            await self._send(f"deck {self.config.deck}")
            self._room_commands_sent = True

        # Wait for guest readiness
        if "loaded a deck" in line and not self._guest_ready:
            self._guest_ready = True
            await self._send("start")

        # RPS transition — menu items identify which menu we're in
        if line == "[1] Rock":
            self.state = State.PRE_DUEL_RPS
            await self._handle_rps(line)

    async def _handle_room_guest(self, line: str) -> None:
        if not self._room_commands_sent:
            await self._send("move 2")
            await self._send(f"deck {self.config.deck}")
            self._room_commands_sent = True

        # RPS transition
        if line == "[1] Rock":
            self.state = State.PRE_DUEL_RPS
            await self._handle_rps(line)

    # -- RPS --

    async def _handle_rps(self, line: str) -> None:
        if line == "[1] Rock":
            choice = str(self._rng.randint(1, 3))
            await self._send(choice)
        elif "you must wait for your opponent" in line:
            # Lost RPS — wait for duel creation
            self._rps_winner = False
        elif line == "[1] Yes":
            # Won RPS — transition to decision
            self._rps_winner = True
            self.state = State.PRE_DUEL_DECISION
            await self._handle_decision(line)
        elif "Duel created." in line:
            self.state = State.DUEL

    # -- Decision (RPS winner only) --

    async def _handle_decision(self, line: str) -> None:
        if line == "[1] Yes":
            await self._send("1")  # Always go first for now
        elif "Duel created." in line:
            self.state = State.DUEL

    # -- Duel --

    async def _handle_duel(self, line: str) -> None:
        if is_duel_end(line):
            # Feed the win/lose event to game state before finishing
            if self._game_state is not None:
                end_result = self._text_parser.feed_line(line)  # type: ignore[union-attr]
                if isinstance(end_result, ParsedEvent):
                    self._game_state.update(end_result)
            logger.info("Duel ended: %s", line)
            self.state = State.FINISHED
            return

        result = self._text_parser.feed_line(line)  # type: ignore[union-attr]

        # Feed events to game state
        if isinstance(result, ParsedEvent) and self._game_state is not None:
            self._game_state.update(result)
            if self.config.verbose:
                logger.info(
                    "[DUEL] event=%s", result.event_type.name)
            # Trigger resync at start of our turn
            if (result.event_type == EventType.NEW_TURN
                    and not result.is_opponent):
                self._resync_pending = True

        if isinstance(result, ParsedPrompt):
            prompt = result
            # Send resync commands before the IDLE_CMD prompt of our turn.
            # Only resync on IDLE_CMD because informational commands (score,
            # h, tab, tab2) cause the server to re-send the active
            # DuelReader prompt — and only the idle re-prompt ("Select a
            # card:") is guaranteed to be in _DUELREADER_REPROMPTS.
            if (self._resync_pending and self._game_state is not None
                    and prompt.prompt_type == PromptType.IDLE_CMD):
                self._resync_pending = False
                await self._send_resync()
                if self.state != State.DUEL:
                    return
            await self._act_on_prompt(prompt)

    async def _act_on_prompt(self, prompt: ParsedPrompt) -> None:
        """Choose an action for *prompt* and send the translated command."""
        # Delegate idle/battle to atomic handlers
        if prompt.prompt_type == PromptType.IDLE_CMD:
            ended = await IdleCmdHandler.handle(
                self.conn, prompt,
                self._agent,  # type: ignore[arg-type]
                self._game_state,
                self._text_parser,  # type: ignore[arg-type]
                self.config.verbose)
            if ended:
                self.state = State.FINISHED
            return

        if prompt.prompt_type == PromptType.BATTLE_MENU:
            ended = await BattleCmdHandler.handle(
                self.conn, prompt,
                self._agent,  # type: ignore[arg-type]
                self._game_state,
                self._text_parser,  # type: ignore[arg-type]
                self.config.verbose)
            if ended:
                self.state = State.FINISHED
            return

        action = self._agent.choose(  # type: ignore[union-attr]
            prompt, game_state=self._game_state)
        text = self._action_translator.translate(action, prompt)  # type: ignore[union-attr]
        if self.config.verbose:
            logger.info(
                "[DUEL] prompt=%s action=%d → %r",
                prompt.prompt_type.name, action, text)
        await self._send(text)

    async def _send_resync(self) -> None:
        """Send resync commands and collect responses.

        Sends ``score``, ``h``, ``tab``, ``tab2``, ``grave``, ``grave2``,
        ``removed``, ``removed2``, ``extra``, ``extra2`` commands.
        The responses arrive as regular lines that are *not* duel prompts
        or events — they are informational text that the parser returns
        as ``None``.  We collect them and feed to the game state resync
        methods.
        """
        for cmd, parser_method, kwargs in [
            ("score", "resync_score", {}),
            ("h", "resync_hand", {}),
            ("tab", "resync_tab", {"opponent": False}),
            ("tab2", "resync_tab", {"opponent": True}),
            ("grave", "resync_grave", {"opponent": False}),
            ("grave2", "resync_grave", {"opponent": True}),
            ("removed", "resync_removed", {"opponent": False}),
            ("removed2", "resync_removed", {"opponent": True}),
            ("extra", "resync_extra", {"opponent": False}),
            ("extra2", "resync_extra", {"opponent": True}),
        ]:
            await self._send(cmd)
            lines: list[str] = []
            # Collect response lines until we get a prompt, event, or
            # empty line that signals end of the command response.
            # The MUD server sends command responses immediately, so
            # we read lines until we see something the parser recognises
            # or a known response terminator.
            while True:
                resp = await self.conn.recv_line()
                if self.config.verbose:
                    logger.info("[RESYNC:%s] recv: %s", cmd, resp)
                # Feed through the parser.  Events can appear in
                # informational responses (e.g. shuffle notifications)
                # — update game state but keep reading.  Only break
                # when the parser produces a prompt (we've overshot).
                check = self._text_parser.feed_line(resp)  # type: ignore[union-attr]
                if isinstance(check, ParsedEvent):
                    if self._game_state is not None:
                        self._game_state.update(check)
                    # Don't break — continue reading the response
                elif isinstance(check, ParsedPrompt):
                    await self._act_on_prompt(check)
                    return
                if is_duel_end(resp):
                    logger.info("Duel ended during resync: %s", resp)
                    self.state = State.FINISHED
                    return
                lines.append(resp)
                # All informational commands end when the server re-sends
                # the active DuelReader prompt ("Select a card:",
                # "Select an option:", "Enter a line of text.").
                if resp in _DUELREADER_REPROMPTS:
                    break
            getattr(self._game_state, parser_method)(lines, **kwargs)

    # -- Helpers --

    async def _send(self, text: str) -> None:
        if self.config.verbose:
            logger.info("[%s] send: %s", self.state.name, text)
        await self.conn.send_line(text)
