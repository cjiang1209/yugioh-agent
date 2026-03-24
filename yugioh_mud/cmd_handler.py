"""Atomic single-decision idle & battle command handlers.

Each handler discovers all available actions, queries the agent ONCE,
then deterministically executes the multi-step MUD conversation.

**Assumption — no interleaved game-state changes during probe:**

The probe phase (``?`` for idle, ``a``/``c`` + ``z`` for battle) sends
informational commands and reads back the server's response.  We assume
the game state does **not** change between the probe and the execution
phase — i.e., the set of legal actions discovered during the probe is
still valid when we execute.  This holds because:

1. It is our turn (the MUD server is blocked waiting for our input).
2. The ``?`` / sub-menu probe commands are purely informational — they
   do not advance the game engine or change any zone contents.
3. The ``z`` (back) command returns us to the original menu without
   side effects.

If a future MUD server version interleaves opponent-triggered events
(e.g. trigger effects that resolve during our input window), the probed
action list could become stale.  In that case the execution phase may
send a cardspec the server no longer considers legal, resulting in an
error or re-prompt.  Handling that would require re-probing after each
interleaved event, which is not implemented.

Similarly, ``card_code`` resolution reads ``game_state`` zones at build
time.  If events were interleaved during the probe and mutated zone
contents (cards moving, being destroyed, etc.), the resolved codes
could be wrong.  The same "no interleaving" assumption prevents this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from yugioh_env.constants import (
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
)
from yugioh_mud.text_parser import (
    DUELREADER_REPROMPTS,
    ParsedEvent, ParsedPrompt, PromptType, _CARDSPEC_LINE_RE,
    is_duel_end,
)

if TYPE_CHECKING:
    from yugioh_mud.agent import Agent
    from yugioh_mud.game_state import MUDGameState
    from yugioh_mud.protocol import Connection
    from yugioh_mud.text_parser import MUDTextParser

logger = logging.getLogger(__name__)

_SPEC_PREFIX_TO_LOCATION = {
    "h": LOCATION_HAND,
    "m": LOCATION_MZONE,
    "s": LOCATION_SZONE,
    "g": LOCATION_GRAVE,
    "x": LOCATION_EXTRA,
    "r": LOCATION_BANISHED,
}

# ---------------------------------------------------------------------------
# StructuredAction
# ---------------------------------------------------------------------------


@dataclass
class StructuredAction:
    """A single atomic action in the idle or battle menu."""
    category: int         # idle: 0-7, battle: 0-3
    cardspec: str = ""    # e.g. "h1", "m2"
    card_code: int = 0    # passcode from game state (0 = unknown/phase)
    location: int = 0     # LOCATION_* constant
    sequence: int = 0     # zone slot (0-indexed)
    sub_action: str = ""  # MUD text for step 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DuelEndedError(Exception):
    """Raised when a duel-end line is received during a handler flow."""


_BARE_SPEC_RE = re.compile(r"^([a-z]+?)(\d+)$")


def parse_cardspec(spec: str) -> tuple[int, int]:
    """Parse a cardspec like 'h1' into (LOCATION_*, 0-indexed sequence).

    Returns (0, 0) if the spec doesn't match a known pattern.
    """
    m = _BARE_SPEC_RE.match(spec)
    if not m:
        return 0, 0
    prefix, digit = m.group(1), int(m.group(2))
    location = _SPEC_PREFIX_TO_LOCATION.get(prefix, 0)
    return location, max(digit - 1, 0)


def _resolve_card_code(
    spec: str, game_state: MUDGameState | None,
) -> int:
    """Look up card_code from game_state zones by cardspec."""
    if game_state is None:
        return 0
    loc, _ = parse_cardspec(spec)
    zone = {
        LOCATION_HAND: game_state.my_hand,
        LOCATION_MZONE: game_state.my_mzone,
        LOCATION_SZONE: game_state.my_szone,
        LOCATION_GRAVE: game_state.my_graveyard,
        LOCATION_BANISHED: game_state.my_banished,
        LOCATION_EXTRA: game_state.my_extra,
    }.get(loc)
    if zone is None:
        return 0
    for card in zone:
        if card.spec == spec:
            return card.code
    return 0


async def _recv_and_process(
    conn: Connection,
    text_parser: MUDTextParser,
    game_state: MUDGameState | None,
    verbose: bool,
) -> tuple[str, object]:
    """Read one line, check duel-end, feed events to game state.

    Returns (raw_line, parser_result).
    Raises DuelEndedError if the line signals duel end.
    """
    line = await conn.recv_line()
    if verbose:
        logger.info("[CMD_HANDLER] recv: %s", line)
    if is_duel_end(line):
        # Feed all events before raising
        for r in text_parser.feed_line_all(line):
            if isinstance(r, ParsedEvent) and game_state is not None:
                game_state.update(r)
        raise DuelEndedError(line)
    results = text_parser.feed_line_all(line)
    prompt_result: object = None
    for r in results:
        if isinstance(r, ParsedEvent) and game_state is not None:
            game_state.update(r)
        elif isinstance(r, ParsedPrompt):
            prompt_result = r
    # Return the prompt if found, otherwise the last result (or None)
    final = prompt_result or (results[-1] if results else None)
    return line, final


async def _send(conn: Connection, text: str, verbose: bool) -> None:
    if verbose:
        logger.info("[CMD_HANDLER] send: %s", text)
    await conn.send_line(text)


# ---------------------------------------------------------------------------
# Idle "?" response category regexes
# ---------------------------------------------------------------------------

class _IdleCategory(NamedTuple):
    regex: re.Pattern[str]
    category: int       # idle action category (0-5)
    sub_action: str     # submenu letter to send

_IDLE_CATEGORIES = [
    _IdleCategory(re.compile(r"^Summonable in attack position:\s*(.+)$"), 0, "s"),
    _IdleCategory(re.compile(r"^Special summonable:\s*(.+)$"), 1, "c"),
    _IdleCategory(re.compile(r"^Repositionable:\s*(.+)$"), 2, "r"),
    _IdleCategory(re.compile(r"^Summonable in defense position:\s*(.+)$"), 3, "m"),
    _IdleCategory(re.compile(r"^Settable:\s*(.+)$"), 4, "t"),
    _IdleCategory(re.compile(r"^Activatable:\s*(.+)$"), 5, "v"),  # see multi-effect TODO below
]


# Submenu letters recognized during execution phase
_SUBMENU_LETTER_RE = re.compile(r"^([a-z]{1,2}): ")

# ---------------------------------------------------------------------------
# IdleCmdHandler
# ---------------------------------------------------------------------------

class IdleCmdHandler:
    """Handles IDLE_CMD prompts atomically: probe → build → decide → execute."""

    @staticmethod
    async def handle(
        conn: Connection,
        prompt: ParsedPrompt,
        agent: Agent,
        game_state: MUDGameState | None,
        text_parser: MUDTextParser,
        verbose: bool,
    ) -> bool:
        """Handle an idle prompt. Returns True if duel ended."""
        try:
            return await IdleCmdHandler._handle_inner(
                conn, prompt, agent, game_state, text_parser, verbose)
        except DuelEndedError as e:
            logger.info("Duel ended during idle handler: %s", e)
            return True

    @staticmethod
    async def _handle_inner(
        conn: Connection,
        prompt: ParsedPrompt,
        agent: Agent,
        game_state: MUDGameState | None,
        text_parser: MUDTextParser,
        verbose: bool,
    ) -> bool:
        # -- Probe phase: send "?" and collect usable cards --
        # The "?" command is informational — it does not advance the game
        # engine.  We assume no interleaved events change zone contents
        # between this probe and the execution phase (see module docstring).
        await _send(conn, "?", verbose)

        # Collect (category, sub_action_letter, cardspec) tuples
        card_actions: list[tuple[int, str, str]] = []
        while True:
            line, result = await _recv_and_process(
                conn, text_parser, game_state, verbose)

            # Check for category lines
            matched = False
            for cat in _IDLE_CATEGORIES:
                m = cat.regex.match(line)
                if m:
                    for spec in m.group(1).split(","):
                        spec = spec.strip()
                        if spec:
                            card_actions.append((cat.category, cat.sub_action, spec))
                    matched = True
                    break

            if matched:
                continue

            # Terminator: DuelReader re-prompt or a new ParsedPrompt
            if line in DUELREADER_REPROMPTS:
                break
            if isinstance(result, ParsedPrompt):
                break

        # -- Build phase: construct StructuredAction list --
        # card_code is resolved from game_state zones at this point.
        # This assumes zone contents haven't drifted since the probe
        # (no interleaved events — see module docstring).
        actions: list[StructuredAction] = []
        for category, sub_letter, spec in card_actions:
            loc, seq = parse_cardspec(spec)
            code = _resolve_card_code(spec, game_state)
            actions.append(StructuredAction(
                category=category,
                cardspec=spec,
                card_code=code,
                location=loc,
                sequence=seq,
                sub_action=sub_letter,
            ))

        # Add phase transitions from original prompt.options
        for opt in prompt.options:
            if opt == "b":
                actions.append(StructuredAction(category=6, sub_action="b"))
            elif opt == "e":
                actions.append(StructuredAction(category=7, sub_action="e"))

        prompt.structured_actions = actions

        # -- Decision phase: query agent once --
        action_idx = agent.choose(prompt, game_state=game_state)

        if verbose:
            logger.info(
                "[IDLE] structured_actions=%d, agent chose=%d",
                len(actions), action_idx)

        # -- Execution phase --
        # Handle END_PHASE meta-command
        from yugioh_mud.agent import END_PHASE
        if action_idx == END_PHASE or action_idx < 0:
            # Send "e" for end phase
            await _send(conn, "e", verbose)
            return False

        if action_idx >= len(actions):
            # Fallback: end phase
            await _send(conn, "e", verbose)
            return False

        chosen = actions[action_idx]

        # Phase transition
        if chosen.category in (6, 7):
            await _send(conn, chosen.sub_action, verbose)
            return False

        # Card action: send cardspec, then wait for submenu
        await _send(conn, chosen.cardspec, verbose)

        # Read lines until submenu arrives ("Select action for " prefix)
        submenu_letters: list[str] = []
        while True:
            line, result = await _recv_and_process(
                conn, text_parser, game_state, verbose)
            # Collect submenu letter lines
            m = _SUBMENU_LETTER_RE.match(line)
            if m:
                submenu_letters.append(m.group(1))
            # Terminal: "Select action for ..."
            if line.startswith("Select action for "):
                break

        # Determine which letter to send
        desired = chosen.sub_action  # e.g. "v", "s", "t", etc.
        if desired in submenu_letters:
            await _send(conn, desired, verbose)
        elif desired == "v":
            # Multi-effect card: the server shows "va", "vb", etc.
            # instead of bare "v".  We always pick the first v* letter.
            #
            # TODO: The agent currently has no way to choose between
            # effects — there is one StructuredAction per activatable
            # cardspec with sub_action="v".  A model agent that cares
            # about which effect to activate will need one
            # StructuredAction per effect (sub_action="va", "vb", …).
            # That requires probing the submenu during the build phase
            # to discover the effect count, which adds a round-trip per
            # activatable card.
            for letter in submenu_letters:
                if letter.startswith("v") and letter != "v":
                    await _send(conn, letter, verbose)
                    break
            else:
                # Fallback: send "z" (back) if desired letter not found
                await _send(conn, "z", verbose)
        else:
            # Fallback: send "z" (back)
            await _send(conn, "z", verbose)

        return False


# ---------------------------------------------------------------------------
# BattleCmdHandler
# ---------------------------------------------------------------------------

class BattleCmdHandler:
    """Handles BATTLE_MENU prompts atomically: probe → build → decide → execute."""

    @staticmethod
    async def handle(
        conn: Connection,
        prompt: ParsedPrompt,
        agent: Agent,
        game_state: MUDGameState | None,
        text_parser: MUDTextParser,
        verbose: bool,
    ) -> bool:
        """Handle a battle prompt. Returns True if duel ended."""
        try:
            return await BattleCmdHandler._handle_inner(
                conn, prompt, agent, game_state, text_parser, verbose)
        except DuelEndedError as e:
            logger.info("Duel ended during battle handler: %s", e)
            return True

    @staticmethod
    async def _handle_inner(
        conn: Connection,
        prompt: ParsedPrompt,
        agent: Agent,
        game_state: MUDGameState | None,
        text_parser: MUDTextParser,
        verbose: bool,
    ) -> bool:
        attack_specs: list[str] = []
        activate_specs: list[str] = []

        # -- Probe phase --
        # We enter each sub-menu ("a" for attack, "c" for activate),
        # collect the available cardspecs, then send "z" to back out.
        # This assumes the game state is frozen during our input window
        # — no interleaved events change the set of legal actions or
        # zone contents between probe and execution (see module docstring).

        # Probe attack cards
        if "a" in prompt.options:
            await _send(conn, "a", verbose)
            # Read until "Select a card:" terminal
            while True:
                line, result = await _recv_and_process(
                    conn, text_parser, game_state, verbose)
                m = _CARDSPEC_LINE_RE.match(line)
                if m:
                    attack_specs.append(m.group(1))
                if line == "Select a card:":
                    break
            # Send "z" (back)
            await _send(conn, "z", verbose)
            # Discard re-sent battle menu until "Select an option:".
            # Read via conn.recv_line() directly — NOT _recv_and_process —
            # because feeding these lines through the text parser would
            # emit a spurious BATTLE_MENU prompt.  This is safe because
            # the server does not interleave events after a "z" back;
            # only the deterministic menu re-send arrives here.
            while True:
                line = await conn.recv_line()
                if verbose:
                    logger.info("[BATTLE_PROBE] discard: %s", line)
                if is_duel_end(line):
                    raise DuelEndedError(line)
                if line == "Select an option:":
                    break

        # Probe activate cards
        if "c" in prompt.options:
            await _send(conn, "c", verbose)
            # Read until "Enter a line of text." terminal
            while True:
                line, result = await _recv_and_process(
                    conn, text_parser, game_state, verbose)
                m = _CARDSPEC_LINE_RE.match(line)
                if m:
                    activate_specs.append(m.group(1))
                if line == "Enter a line of text.":
                    break
            # Send "z" (back) — same discard rationale as above.
            await _send(conn, "z", verbose)
            while True:
                line = await conn.recv_line()
                if verbose:
                    logger.info("[BATTLE_PROBE] discard: %s", line)
                if is_duel_end(line):
                    raise DuelEndedError(line)
                if line == "Select an option:":
                    break

        # -- Build phase --
        # card_code is resolved from game_state zones at this point.
        # This assumes zone contents haven't drifted since the probe
        # (no interleaved events — see module docstring).
        actions: list[StructuredAction] = []

        for spec in activate_specs:
            loc, seq = parse_cardspec(spec)
            code = _resolve_card_code(spec, game_state)
            actions.append(StructuredAction(
                category=0, cardspec=spec, card_code=code,
                location=loc, sequence=seq, sub_action=spec,
            ))

        for spec in attack_specs:
            loc, seq = parse_cardspec(spec)
            code = _resolve_card_code(spec, game_state)
            actions.append(StructuredAction(
                category=1, cardspec=spec, card_code=code,
                location=loc, sequence=seq, sub_action=spec,
            ))

        if "m" in prompt.options:
            actions.append(StructuredAction(category=2, sub_action="m"))
        if "e" in prompt.options:
            actions.append(StructuredAction(category=3, sub_action="e"))

        prompt.structured_actions = actions

        # -- Decision phase --
        action_idx = agent.choose(prompt, game_state=game_state)

        if verbose:
            logger.info(
                "[BATTLE] structured_actions=%d, agent chose=%d",
                len(actions), action_idx)

        # -- Execution phase --
        from yugioh_mud.agent import END_PHASE
        if action_idx == END_PHASE or action_idx < 0:
            await _send(conn, "e", verbose)
            return False

        if action_idx >= len(actions):
            await _send(conn, "e", verbose)
            return False

        chosen = actions[action_idx]

        # Phase transitions
        if chosen.category in (2, 3):
            await _send(conn, chosen.sub_action, verbose)
            return False

        # Attack
        if chosen.category == 1:
            await _send(conn, "a", verbose)
            # Wait for "Select a card:" (BATTLE_SELECT)
            while True:
                line, result = await _recv_and_process(
                    conn, text_parser, game_state, verbose)
                if line == "Select a card:":
                    break
            # Send the attacker cardspec
            await _send(conn, chosen.cardspec, verbose)
            # Target selection handled by normal prompt loop
            return False

        # Activate
        if chosen.category == 0:
            await _send(conn, "c", verbose)
            # Wait for "Enter a line of text." (BATTLE_SELECT)
            while True:
                line, result = await _recv_and_process(
                    conn, text_parser, game_state, verbose)
                if line == "Enter a line of text.":
                    break
            # Send the cardspec
            await _send(conn, chosen.cardspec, verbose)
            return False

        # Fallback
        await _send(conn, "e", verbose)
        return False
