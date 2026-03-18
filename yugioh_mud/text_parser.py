"""MUD duel line classifier, prompt parser, and event parser.

Classifies incoming MUD server lines into prompt types (requiring a
response) or events (informational state changes).  The parser is a
line-oriented state machine with two modes: scanning (looking for prompt
headers) and accumulating (collecting numbered options until a known
terminal line arrives).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yugioh_mud.cmd_handler import StructuredAction


# ---------------------------------------------------------------------------
# Public types — events (informational lines)
# ---------------------------------------------------------------------------

class EventType(Enum):
    NEW_TURN = auto()
    NEW_PHASE = auto()
    DAMAGE = auto()
    RECOVER = auto()
    PAY_LP = auto()
    DRAW = auto()
    SUMMON = auto()
    SP_SUMMON = auto()
    FLIP_SUMMON = auto()
    SET = auto()
    POS_CHANGE = auto()
    ATTACK = auto()
    CHAINING = auto()
    DESTROY = auto()
    TO_GRAVEYARD = auto()
    BANISHED = auto()
    TO_HAND = auto()
    TO_DECK = auto()
    TO_EXTRA_DECK = auto()
    TRIBUTE = auto()
    DISCARD = auto()
    EQUIP = auto()
    SHUFFLE = auto()
    WIN = auto()
    LOSE = auto()


@dataclass
class ParsedEvent:
    """Structured informational event from a MUD server line."""
    event_type: EventType
    player: str = ""        # "you" or opponent nickname
    is_opponent: bool = False
    card_name: str = ""
    card_spec: str = ""     # zone spec e.g. "m1", "s2", "oh3"
    target_spec: str = ""
    target_name: str = ""
    amount: int = 0         # LP change, draw count, etc.
    new_lp: int = 0
    position: str = ""      # "face-up attack", "face-down defense", etc.
    phase: str = ""         # phase string for NEW_PHASE
    raw: str = ""


# ---------------------------------------------------------------------------
# Public types — prompts (require a response)
# ---------------------------------------------------------------------------

class PromptType(Enum):
    IDLE_CMD = auto()
    IDLE_SUBMENU = auto()
    BATTLE_MENU = auto()
    BATTLE_SELECT = auto()
    SELECT_CARD = auto()
    SELECT_TRIBUTE = auto()
    SELECT_CHAIN = auto()
    SELECT_EFFECTYN = auto()
    SELECT_YESNO = auto()
    SELECT_POSITION = auto()
    SELECT_PLACE = auto()
    SELECT_OPTION = auto()
    SELECT_SUM = auto()
    SELECT_COUNTER = auto()
    SELECT_UNSELECT = auto()
    ANNOUNCE_RACE = auto()
    ANNOUNCE_ATTRIB = auto()
    ANNOUNCE_NUMBER = auto()
    ANNOUNCE_CARD = auto()
    SORT_CARD = auto()
    UNKNOWN = auto()


@dataclass
class ParsedPrompt:
    prompt_type: PromptType
    options: list[str] = field(default_factory=list)
    min_select: int = 1
    max_select: int = 1
    cancelable: bool = False
    finishable: bool = False
    raw_lines: list[str] = field(default_factory=list)
    structured_actions: list[StructuredAction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Duel-end detection
# ---------------------------------------------------------------------------

DUEL_END_PATTERNS = ("You won", "You lost", "You scooped", "was cancelled")


def is_duel_end(line: str) -> bool:
    """Return True if *line* signals the end of a duel."""
    return any(p in line for p in DUEL_END_PATTERNS)


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

_SELECT_CARD_RE = re.compile(
    r"^Select (\d+) to (\d+) cards separated by spaces:")
_SELECT_TRIBUTE_RE = re.compile(
    r"^Select (\d+) to (\d+) cards to tribute separated by spaces:")
_SELECT_PLACE_ONE_RE = re.compile(
    r"^Select place for card, one of (.+)\.$")
_SELECT_PLACE_MULTI_RE = re.compile(
    r"^Select (\d+) places for card, from (.+)\.$")
_SELECT_SUM_RE = re.compile(r"^Select cards with a total value")
_COUNTER_RE = re.compile(r"^Type new .+ for (\d+) cards")
_UNSELECT_RE = re.compile(r"^Check or uncheck (\d+) to (\d+) cards")
_RACE_RE = re.compile(r"^Type (\d+) races? separated by spaces")
_ATTRIB_RE = re.compile(r"^Type (\d+) attributes? separated by spaces")
_SORT_RE = re.compile(r"^Sort (\d+) cards by entering")
_ANNOUNCE_NUM_RE = re.compile(r"^Select a number, one of: (.+)")

# Option line patterns
_OPTION_LINE_RE = re.compile(r"^(\d+): (.+)")
_MENU_LINE_RE = re.compile(r"^\[(\d+)\] (.+)")

# Idle/battle letter command and cardspec patterns
_LETTER_CMD_RE = re.compile(r"^([a-z]{1,2}): (.+)")
_CARDSPEC_LINE_RE = re.compile(r"^([a-z]+\d+): (.+)")

# Known DuelReader terminal (Reader.explain with prompt=None, no_abort set)
_ENTER_TEXT = "Enter a line of text."
# Default DuelMenu terminal
_TYPE_NUMBER = "Type a number or @abort to abort."

# ---------------------------------------------------------------------------
# Event regexes (informational lines → ParsedEvent)
# ---------------------------------------------------------------------------

# Turn: "Your turn." / "{nick}'s turn."
_YOUR_TURN_RE = re.compile(r"^Your turn\.$")
_OPP_TURN_RE = re.compile(r"^(.+?)'s turn\.$")

# Phase: "entering {phase}."
_PHASE_RE = re.compile(r"^entering (.+?)\.$")

# LP — damage: "Your lp decreased by {n}, now {n}" /
#               "{nick}'s lp decreased by {n}, now {n}"
_YOUR_DAMAGE_RE = re.compile(
    r"^Your lp decreased by (\d+), now (\d+)$")
_OPP_DAMAGE_RE = re.compile(
    r"^(.+?)'s lp decreased by (\d+), now (\d+)$")

# LP — recover: "Your lp increased by {n}, now {n}" /
#               "{nick}'s lp increased by {n}, now {n}"
_YOUR_RECOVER_RE = re.compile(
    r"^Your lp increased by (\d+), now (\d+)$")
_OPP_RECOVER_RE = re.compile(
    r"^(.+?)'s lp increased by (\d+), now (\d+)$")

# LP — pay cost: "You pay {n} LP. Your LP is now {n}." /
#                "{nick} pays {n} LP. Their LP is now {n}."
_YOUR_PAY_LP_RE = re.compile(
    r"^You pay (\d+) LP\. Your LP is now (\d+)\.$")
_OPP_PAY_LP_RE = re.compile(
    r"^(.+?) pays (\d+) LP\. Their LP is now (\d+)\.$")

# Draw: "Drew {n} cards:" / "Opponent drew {n} cards." /
#        "{nick} drew {n} cards."
_YOUR_DRAW_RE = re.compile(r"^Drew (\d+) cards?:")
_OPP_DRAW_RE = re.compile(r"^(?:Opponent|(.+?)) drew (\d+) cards?\.$")

# Summon: "{nick} summoning {card} ({atk}/{def}) in {pos} position."
_SUMMON_RE = re.compile(
    r"^(.+?) summoning (.+?) \((\d+)/(\d+)\) in (.+?) position\.$")

# Special summon: "{nick} special summoning {card} ({atk}/{def}) in {pos} position."
#   or link: "{nick} special summoning {card} ({atk}) in {pos} position."
_SP_SUMMON_RE = re.compile(
    r"^(.+?) special summoning (.+?) \((\d+)(?:/\d+)?\) in (.+?) position\.$")

# Flip summon: "{player} flip summons {card} ({spec})."
_FLIP_SUMMON_RE = re.compile(
    r"^(.+?) flip summons (.+?) \((.+?)\)\.$")

# Set (self): "You set {spec} ({card}) in {pos} position."
# Set (opp):  "{nick} sets {spec} in {pos} position."
_YOUR_SET_RE = re.compile(
    r"^You set (.+?) \((.+?)\) in (.+?) position\.$")
_OPP_SET_RE = re.compile(
    r"^(.+?) sets (.+?) in (.+?) position\.$")

# Position change: "The position of card {spec} ({card}) was changed to {pos}."
_POS_CHANGE_RE = re.compile(
    r"^The position of card (.+?) \((.+?)\) was changed to (.+?)\.$")

# Attack (targeted): "{nick} prepares to attack {tspec} ({tname}) with {spec} ({name})"
# Attack (direct):   "{nick} prepares to attack with {spec} ({name})"
_ATTACK_TARGET_RE = re.compile(
    r"^(.+?) prepares to attack (.+?) \((.+?)\) with (.+?) \((.+?)\)$")
_ATTACK_DIRECT_RE = re.compile(
    r"^(.+?) prepares to attack with (.+?) \((.+?)\)$")

# Chaining: "Activating {spec} ({card})" / "{nick} activating {spec} ({card})"
_YOUR_CHAIN_RE = re.compile(r"^Activating (.+?) \((.+?)\)$")
_OPP_CHAIN_RE = re.compile(r"^(.+?) activating (.+?) \((.+?)\)$")

# Destroy: "Card {spec} ({name}) destroyed."
_DESTROY_RE = re.compile(r"^Card (.+?) \((.+?)\) destroyed\.$")

# To graveyard (self): "your card {spec} ({name}) was sent to the graveyard."
# To graveyard (opp):  "{nick}'s card {spec} ({name}) was sent to the graveyard."
_YOUR_TO_GY_RE = re.compile(
    r"^your card (.+?) \((.+?)\) was sent to the graveyard\.$")
_OPP_TO_GY_RE = re.compile(
    r"^(.+?)'s card (.+?) \((.+?)\) was sent to the graveyard\.$")

# Banished (self): "your card {spec} ({name}) was banished."
# Banished (opp):  "{nick}'s card {spec} ({name}) was banished."
_YOUR_BANISHED_RE = re.compile(
    r"^your card (.+?) \((.+?)\) was banished\.$")
_OPP_BANISHED_RE = re.compile(
    r"^(.+?)'s card (.+?) \((.+?)\) was banished\.$")

# Return to hand: "Card {spec} ({name}) returned to hand." (self)
#                 "{nick}'s card {spec} ({name}) returned to their hand." (opp)
_YOUR_TO_HAND_RE = re.compile(
    r"^Card (.+?) \((.+?)\) returned to hand\.$")
_OPP_TO_HAND_RE = re.compile(
    r"^(.+?)'s card (.+?) \((.+?)\) returned to their hand\.$")

# Return to deck (self): "your card {spec} ({name}) returned to your deck."
# Return to deck (opp):  "{nick}'s card {spec} ({name}) returned to their deck."
_YOUR_TO_DECK_RE = re.compile(
    r"^your card (.+?) \((.+?)\) returned to your deck\.$")
_OPP_TO_DECK_RE = re.compile(
    r"^(.+?)'s card (.+?) \((.+?)\) returned to their deck\.$")

# Return to extra deck (self): "your card {spec} ({name}) returned to your extra deck."
# Return to extra deck (opp):  "{nick}'s card {spec} ({name}) returned to their extra deck."
_YOUR_TO_EXTRA_RE = re.compile(
    r"^your card (.+?) \((.+?)\) returned to your extra deck\.$")
_OPP_TO_EXTRA_RE = re.compile(
    r"^(.+?)'s card (.+?) \((.+?)\) returned to their extra deck\.$")

# Tribute (self): "You tribute {spec} ({name})."
# Tribute (opp):  "{nick} tributes {spec} ({name})."
_YOUR_TRIBUTE_RE = re.compile(r"^You tribute (.+?) \((.+?)\)\.$")
_OPP_TRIBUTE_RE = re.compile(r"^(.+?) tributes (.+?) \((.+?)\)\.$")

# Discard (self): "you discarded {spec} ({name})."
# Discard (opp):  "{nick} discarded {spec} ({name})."
_YOUR_DISCARD_RE = re.compile(r"^you discarded (.+?) \((.+?)\)\.$")
_OPP_DISCARD_RE = re.compile(r"^(.+?) discarded (.+?) \((.+?)\)\.$")

# Equip: "{card} equipped to {target}."
_EQUIP_RE = re.compile(r"^(.+?) equipped to (.+?)\.$")

# Shuffle: "you shuffled your deck." / "{nick} shuffled their deck."
_YOUR_SHUFFLE_RE = re.compile(r"^you shuffled your deck\.$")
_OPP_SHUFFLE_RE = re.compile(r"^(.+?) shuffled their deck\.$")

# Win/Lose: "You won ({reason})." / "You lost ({reason})."
_WIN_RE = re.compile(r"^You won \((.+?)\)\.$")
_LOSE_RE = re.compile(r"^You lost \((.+?)\)\.$")


# ---------------------------------------------------------------------------
# Parser state machine
# ---------------------------------------------------------------------------

class _Mode(Enum):
    SCANNING = auto()
    ACCUMULATING = auto()


class MUDTextParser:
    """Line-oriented parser for MUD duel prompts."""

    def __init__(self) -> None:
        self._mode = _Mode.SCANNING
        self._pending_type: PromptType | None = None
        self._options: list[str] = []
        self._raw_lines: list[str] = []
        self._min_select = 1
        self._max_select = 1
        self._cancelable = False
        self._finishable = False
        self._terminal: str | None = None
        self._terminal_is_prefix: bool = False
        # Context flags
        self._idle_context = False
        self._battle_context = False

    def reset(self) -> None:
        """Reset all parser state."""
        self.__init__()  # type: ignore[misc]

    def feed_line(self, line: str) -> ParsedPrompt | ParsedEvent | None:
        """Feed a single line from the server.

        Returns a ``ParsedPrompt`` when a complete prompt has been detected,
        a ``ParsedEvent`` for informational state-change lines, or ``None``
        for unrecognised / non-terminal lines.

        Lines containing embedded newlines (some MUD server messages pack
        multiple logical lines into one WebSocket frame) are split and
        processed individually; the first resulting prompt or event (if any)
        is returned.
        """
        # Handle embedded newlines from single WebSocket frames
        if "\n" in line:
            result: ParsedPrompt | ParsedEvent | None = None
            for sub in line.split("\n"):
                sub = sub.strip()
                if sub:
                    r = self._feed_single(sub)
                    if r is not None and result is None:
                        result = r
            return result
        return self._feed_single(line)

    def _feed_single(self, line: str) -> ParsedPrompt | ParsedEvent | None:
        if self._mode == _Mode.ACCUMULATING:
            return self._accumulate(line)
        return self._scan(line)

    # -- Accumulation helpers -----------------------------------------------

    def _start_accum(
        self,
        ptype: PromptType,
        terminal: str,
        *,
        min_sel: int = 1,
        max_sel: int = 1,
        cancelable: bool = False,
        finishable: bool = False,
        terminal_is_prefix: bool = False,
    ) -> None:
        self._mode = _Mode.ACCUMULATING
        self._pending_type = ptype
        self._terminal = terminal
        self._terminal_is_prefix = terminal_is_prefix
        self._min_select = min_sel
        self._max_select = max_sel
        self._cancelable = cancelable
        self._finishable = finishable
        self._options = []
        self._raw_lines = []

    def _finalize(self) -> ParsedPrompt:
        prompt = ParsedPrompt(
            prompt_type=self._pending_type,  # type: ignore[arg-type]
            options=list(self._options),
            min_select=self._min_select,
            max_select=self._max_select,
            cancelable=self._cancelable,
            finishable=self._finishable,
            raw_lines=list(self._raw_lines),
        )
        self._mode = _Mode.SCANNING
        self._pending_type = None
        self._options = []
        self._raw_lines = []
        self._min_select = 1
        self._max_select = 1
        self._cancelable = False
        self._finishable = False
        self._terminal = None
        self._terminal_is_prefix = False
        return prompt

    # -- Scanning mode ------------------------------------------------------

    def _scan(self, line: str) -> ParsedPrompt | None:  # noqa: C901
        # --- Context-setting headers ---

        if line == "Select a card on which to perform an action.":
            self._idle_context = True
            self._battle_context = False
            self._start_accum(PromptType.IDLE_CMD, "Select a card:")
            self._raw_lines.append(line)
            return None

        if line == "Battle menu:":
            self._battle_context = True
            self._idle_context = False
            self._start_accum(PromptType.BATTLE_MENU, "Select an option:")
            self._raw_lines.append(line)
            return None

        # --- Single-line prompts (return immediately) ---

        if line.startswith("Do you want to use the effect from"):
            return ParsedPrompt(PromptType.SELECT_EFFECTYN, raw_lines=[line])

        m = _ANNOUNCE_NUM_RE.match(line)
        if m:
            nums = [s.strip() for s in m.group(1).split(",")]
            return ParsedPrompt(
                PromptType.ANNOUNCE_NUMBER, options=nums, raw_lines=[line])

        # Idle submenu: action letters arrive *before* the terminal line
        # "Select action for {name}".  When we're in idle context and see
        # a known submenu letter command, start accumulation.
        if self._idle_context:
            m = _LETTER_CMD_RE.match(line)
            if m:
                letter = m.group(1)
                _SUBMENU_LETTERS = {"s", "t", "m", "r", "c", "v", "z"}
                if letter in _SUBMENU_LETTERS or letter.startswith("v"):
                    self._idle_context = False
                    self._start_accum(
                        PromptType.IDLE_SUBMENU,
                        "Select action for ",
                        terminal_is_prefix=True)
                    self._raw_lines.append(line)
                    self._options.append(letter)
                    return None

        # Idle submenu terminal (DuelReader prompt for act_on_card) —
        # if we reach here without accumulation, return immediately.
        if line.startswith("Select action for "):
            self._idle_context = False
            return ParsedPrompt(PromptType.IDLE_SUBMENU, raw_lines=[line])

        # --- Multi-line prompt headers ---

        # Select place (single header line contains all specs)
        m = _SELECT_PLACE_ONE_RE.match(line)
        if m:
            specs = [s.strip() for s in m.group(1).split(",")]
            self._start_accum(PromptType.SELECT_PLACE, _ENTER_TEXT)
            self._options = specs
            self._raw_lines.append(line)
            return None

        m = _SELECT_PLACE_MULTI_RE.match(line)
        if m:
            count = int(m.group(1))
            specs = [s.strip() for s in m.group(2).split(",")]
            self._start_accum(
                PromptType.SELECT_PLACE, _ENTER_TEXT,
                min_sel=count, max_sel=count)
            self._options = specs
            self._raw_lines.append(line)
            return None

        # Select tribute (check before select_card to avoid ambiguity)
        m = _SELECT_TRIBUTE_RE.match(line)
        if m:
            self._start_accum(
                PromptType.SELECT_TRIBUTE, _ENTER_TEXT,
                min_sel=int(m.group(1)), max_sel=int(m.group(2)))
            self._raw_lines.append(line)
            return None

        m = _SELECT_CARD_RE.match(line)
        if m:
            self._start_accum(
                PromptType.SELECT_CARD, _ENTER_TEXT,
                min_sel=int(m.group(1)), max_sel=int(m.group(2)))
            self._raw_lines.append(line)
            return None

        # Select chain
        if line == "Select chain:":
            self._start_accum(
                PromptType.SELECT_CHAIN, "Select card to chain:")
            self._raw_lines.append(line)
            return None

        if line == "Select chain (c to cancel):":
            self._start_accum(
                PromptType.SELECT_CHAIN,
                "Select card to chain (c = cancel):",
                cancelable=True)
            self._raw_lines.append(line)
            return None

        # Select position (DuelMenu)
        if line.startswith("Select position for"):
            self._start_accum(PromptType.SELECT_POSITION, _TYPE_NUMBER)
            self._raw_lines.append(line)
            return None

        # Select option (DuelMenu — title and prompt are both "Select option:")
        if line == "Select option:":
            self._start_accum(PromptType.SELECT_OPTION, "Select option:")
            self._raw_lines.append(line)
            return None

        # Select sum
        m = _SELECT_SUM_RE.match(line)
        if m:
            self._start_accum(PromptType.SELECT_SUM, _ENTER_TEXT)
            self._raw_lines.append(line)
            return None

        # Select counter
        m = _COUNTER_RE.match(line)
        if m:
            count = int(m.group(1))
            self._start_accum(
                PromptType.SELECT_COUNTER, _ENTER_TEXT,
                min_sel=count, max_sel=count)
            self._raw_lines.append(line)
            return None

        # Select unselect card
        m = _UNSELECT_RE.match(line)
        if m:
            self._start_accum(
                PromptType.SELECT_UNSELECT, _ENTER_TEXT,
                min_sel=int(m.group(1)), max_sel=int(m.group(2)))
            self._raw_lines.append(line)
            return None

        # Announce race
        m = _RACE_RE.match(line)
        if m:
            n = int(m.group(1))
            self._start_accum(
                PromptType.ANNOUNCE_RACE, _ENTER_TEXT,
                min_sel=n, max_sel=n)
            self._raw_lines.append(line)
            return None

        # Announce attrib
        m = _ATTRIB_RE.match(line)
        if m:
            n = int(m.group(1))
            self._start_accum(
                PromptType.ANNOUNCE_ATTRIB, _ENTER_TEXT,
                min_sel=n, max_sel=n)
            self._raw_lines.append(line)
            return None

        # Sort card
        m = _SORT_RE.match(line)
        if m:
            n = int(m.group(1))
            self._start_accum(
                PromptType.SORT_CARD, _ENTER_TEXT,
                min_sel=n, max_sel=n, cancelable=True)
            self._raw_lines.append(line)
            return None

        # Announce card
        if line == "Enter the name of a card:":
            self._start_accum(PromptType.ANNOUNCE_CARD, _ENTER_TEXT)
            self._raw_lines.append(line)
            return None

        # Battle sub-prompts
        if line == "Select card to attack with:":
            self._start_accum(PromptType.BATTLE_SELECT, "Select a card:")
            self._raw_lines.append(line)
            return None

        if line == "Select card to activate:":
            self._start_accum(PromptType.BATTLE_SELECT, _ENTER_TEXT)
            self._raw_lines.append(line)
            return None

        # Fallback: DuelMenu item in scanning mode (unrecognized header)
        m = _MENU_LINE_RE.match(line)
        if m:
            # Most likely a DuelMenu whose title we missed.
            # Try both known terminals; default to TYPE_NUMBER.
            self._start_accum(PromptType.SELECT_OPTION, _TYPE_NUMBER)
            self._options.append(m.group(2))
            self._raw_lines.append(line)
            return None

        # MSG_YESNO: single-line yes/no question from the engine (replay,
        # direct attack, normal summon without tribute, etc.).  The server
        # sends the question text and waits for y/n — no follow-up lines.
        # Checked late so more specific patterns (effectyn, announce) match
        # first.
        if line.endswith("?"):
            return ParsedPrompt(PromptType.SELECT_YESNO, raw_lines=[line])

        # --- Event parsing (informational lines) ---
        return self._scan_event(line)

    # -- Event parsing (informational lines) ---------------------------------

    def _scan_event(self, line: str) -> ParsedEvent | None:  # noqa: C901
        """Try to parse *line* as an informational event."""

        # -- Turn --
        if _YOUR_TURN_RE.match(line):
            return ParsedEvent(EventType.NEW_TURN, player="you", raw=line)
        m = _OPP_TURN_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.NEW_TURN, player=m.group(1),
                is_opponent=True, raw=line)

        # -- Phase --
        m = _PHASE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.NEW_PHASE, phase=m.group(1), raw=line)

        # -- LP: damage --
        m = _YOUR_DAMAGE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DAMAGE, player="you",
                amount=int(m.group(1)), new_lp=int(m.group(2)), raw=line)
        m = _OPP_DAMAGE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DAMAGE, player=m.group(1), is_opponent=True,
                amount=int(m.group(2)), new_lp=int(m.group(3)), raw=line)

        # -- LP: recover --
        m = _YOUR_RECOVER_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.RECOVER, player="you",
                amount=int(m.group(1)), new_lp=int(m.group(2)), raw=line)
        m = _OPP_RECOVER_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.RECOVER, player=m.group(1), is_opponent=True,
                amount=int(m.group(2)), new_lp=int(m.group(3)), raw=line)

        # -- LP: pay cost --
        m = _YOUR_PAY_LP_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.PAY_LP, player="you",
                amount=int(m.group(1)), new_lp=int(m.group(2)), raw=line)
        m = _OPP_PAY_LP_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.PAY_LP, player=m.group(1), is_opponent=True,
                amount=int(m.group(2)), new_lp=int(m.group(3)), raw=line)

        # -- Draw --
        m = _YOUR_DRAW_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DRAW, player="you",
                amount=int(m.group(1)), raw=line)
        m = _OPP_DRAW_RE.match(line)
        if m:
            nick = m.group(1) or "Opponent"
            return ParsedEvent(
                EventType.DRAW, player=nick, is_opponent=True,
                amount=int(m.group(2)), raw=line)

        # -- Special summon (check before normal summon) --
        m = _SP_SUMMON_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.SP_SUMMON, player=m.group(1),
                card_name=m.group(2), position=m.group(4), raw=line)

        # -- Summon (normal) --
        m = _SUMMON_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.SUMMON, player=m.group(1),
                card_name=m.group(2), position=m.group(5), raw=line)

        # -- Flip summon --
        m = _FLIP_SUMMON_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.FLIP_SUMMON, player=m.group(1),
                card_name=m.group(2), card_spec=m.group(3), raw=line)

        # -- Set --
        m = _YOUR_SET_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.SET, player="you", card_spec=m.group(1),
                card_name=m.group(2), position=m.group(3), raw=line)
        m = _OPP_SET_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.SET, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), position=m.group(3), raw=line)

        # -- Position change --
        m = _POS_CHANGE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.POS_CHANGE, card_spec=m.group(1),
                card_name=m.group(2), position=m.group(3), raw=line)

        # -- Attack --
        m = _ATTACK_TARGET_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.ATTACK, player=m.group(1),
                card_spec=m.group(4), card_name=m.group(5),
                target_spec=m.group(2), target_name=m.group(3), raw=line)
        m = _ATTACK_DIRECT_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.ATTACK, player=m.group(1),
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Chaining --
        m = _YOUR_CHAIN_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.CHAINING, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_CHAIN_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.CHAINING, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Destroy --
        m = _DESTROY_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DESTROY, card_spec=m.group(1),
                card_name=m.group(2), raw=line)

        # -- To graveyard --
        m = _YOUR_TO_GY_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_GRAVEYARD, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_TO_GY_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_GRAVEYARD, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Banished --
        m = _YOUR_BANISHED_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.BANISHED, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_BANISHED_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.BANISHED, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- To hand --
        m = _YOUR_TO_HAND_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_HAND, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_TO_HAND_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_HAND, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- To deck --
        m = _YOUR_TO_DECK_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_DECK, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_TO_DECK_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_DECK, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- To extra deck --
        m = _YOUR_TO_EXTRA_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_EXTRA_DECK, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_TO_EXTRA_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TO_EXTRA_DECK, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Tribute --
        m = _YOUR_TRIBUTE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TRIBUTE, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_TRIBUTE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.TRIBUTE, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Discard --
        m = _YOUR_DISCARD_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DISCARD, player="you",
                card_spec=m.group(1), card_name=m.group(2), raw=line)
        m = _OPP_DISCARD_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.DISCARD, player=m.group(1), is_opponent=True,
                card_spec=m.group(2), card_name=m.group(3), raw=line)

        # -- Equip --
        m = _EQUIP_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.EQUIP, card_name=m.group(1),
                target_name=m.group(2), raw=line)

        # -- Shuffle --
        if _YOUR_SHUFFLE_RE.match(line):
            return ParsedEvent(
                EventType.SHUFFLE, player="you", raw=line)
        m = _OPP_SHUFFLE_RE.match(line)
        if m:
            return ParsedEvent(
                EventType.SHUFFLE, player=m.group(1),
                is_opponent=True, raw=line)

        # -- Win / Lose --
        m = _WIN_RE.match(line)
        if m:
            return ParsedEvent(EventType.WIN, player="you", raw=line)
        m = _LOSE_RE.match(line)
        if m:
            return ParsedEvent(EventType.LOSE, player="you", raw=line)

        # Truly unrecognised
        return None

    # -- Accumulation mode --------------------------------------------------

    def _accumulate(self, line: str) -> ParsedPrompt | None:
        self._raw_lines.append(line)

        # Terminal check (exact or prefix)
        if self._terminal is not None:
            if self._terminal_is_prefix:
                if line.startswith(self._terminal):
                    return self._finalize()
            elif line == self._terminal:
                return self._finalize()

        # For SELECT_OPTION accumulated via missed title, also accept
        # "Select option:" as an alternative terminal.
        if (self._pending_type == PromptType.SELECT_OPTION
                and self._terminal == _TYPE_NUMBER
                and line == "Select option:"):
            return self._finalize()

        # Numbered option lines ("N: text")
        m = _OPTION_LINE_RE.match(line)
        if m:
            self._options.append(m.group(2))
            return None

        # DuelMenu item lines ("[N] text")
        m = _MENU_LINE_RE.match(line)
        if m:
            self._options.append(m.group(2))
            return None

        # Chain spec options ("m1: Card Name" or "m1a (Card): desc")
        if self._pending_type == PromptType.SELECT_CHAIN and ":" in line:
            spec = line.split(":")[0].split("(")[0].strip()
            if spec:
                self._options.append(spec)
            return None

        # Detect finishable/cancelable flags in accumulation
        if line == "Enter f to finish":
            self._finishable = True
            return None
        if line == "Enter c to cancel":
            self._cancelable = True
            return None

        # IDLE_CMD / BATTLE_MENU: extract letter commands into options
        if self._pending_type in (PromptType.IDLE_CMD,
                                  PromptType.BATTLE_MENU):
            m = _LETTER_CMD_RE.match(line)
            if m:
                self._options.append(m.group(1))
                return None

        # IDLE_SUBMENU: extract letter commands (skip 'i' for info)
        if self._pending_type == PromptType.IDLE_SUBMENU:
            m = _LETTER_CMD_RE.match(line)
            if m and m.group(1) != "i":
                self._options.append(m.group(1))
                return None

        # BATTLE_SELECT: extract cardspecs and 'z' back
        if self._pending_type == PromptType.BATTLE_SELECT:
            m = _CARDSPEC_LINE_RE.match(line)
            if m:
                self._options.append(m.group(1))
                return None
            m = _LETTER_CMD_RE.match(line)
            if m:
                self._options.append(m.group(1))
                return None

        return None
