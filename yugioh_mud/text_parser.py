"""MUD duel line classifier and prompt parser.

Classifies incoming MUD server lines into prompt types and extracts enough
structure for automated play.  The parser is a line-oriented state machine
with two modes: scanning (looking for prompt headers) and accumulating
(collecting numbered options until a known terminal line arrives).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Public types
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

# Known DuelReader terminal (Reader.explain with prompt=None, no_abort set)
_ENTER_TEXT = "Enter a line of text."
# Default DuelMenu terminal
_TYPE_NUMBER = "Type a number or @abort to abort."


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
        # Context flags
        self._idle_context = False
        self._battle_context = False

    def reset(self) -> None:
        """Reset all parser state."""
        self.__init__()  # type: ignore[misc]

    def feed_line(self, line: str) -> ParsedPrompt | None:
        """Feed a single line from the server.

        Returns a ``ParsedPrompt`` when a complete prompt has been detected,
        or ``None`` for informational / non-terminal lines.

        Lines containing embedded newlines (some MUD server messages pack
        multiple logical lines into one WebSocket frame) are split and
        processed individually; the first resulting prompt (if any) is
        returned.
        """
        # Handle embedded newlines from single WebSocket frames
        if "\n" in line:
            result: ParsedPrompt | None = None
            for sub in line.split("\n"):
                sub = sub.strip()
                if sub:
                    r = self._feed_single(sub)
                    if r is not None and result is None:
                        result = r
            return result
        return self._feed_single(line)

    def _feed_single(self, line: str) -> ParsedPrompt | None:
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
    ) -> None:
        self._mode = _Mode.ACCUMULATING
        self._pending_type = ptype
        self._terminal = terminal
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

        # Idle submenu terminal (DuelReader prompt for act_on_card)
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

        # Nothing matched — informational line
        return None

    # -- Accumulation mode --------------------------------------------------

    def _accumulate(self, line: str) -> ParsedPrompt | None:
        self._raw_lines.append(line)

        # Terminal check
        if self._terminal is not None and line == self._terminal:
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

        # Idle/battle option lines ("letter: description")
        # Just accumulate as raw lines (no options extraction needed for
        # passive agent — it uses fixed letter commands).
        return None
