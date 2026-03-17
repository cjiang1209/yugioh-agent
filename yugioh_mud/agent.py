"""Duel agents for the MUD bot.

An ``Agent`` chooses an action given a parsed prompt.  The
``ActionTranslator`` (see ``action_translator.py``) converts the int
return value into the MUD text command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from yugioh_mud.text_parser import ParsedPrompt, PromptType

if TYPE_CHECKING:
    from yugioh_mud.game_state import MUDGameState

# ---------------------------------------------------------------------------
# Special action constants (negative = meta-commands)
# ---------------------------------------------------------------------------

END_PHASE = -1
CANCEL = -2
DECLINE = -3
BACK = -4
FINISH = -5


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------

class Agent(Protocol):
    """Decides *what* to do given a parsed prompt.

    Returns an int:
    * ``>= 0`` — index into ``prompt.options`` (0-based)
    * negative — one of the special constants above
    """

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int: ...


# ---------------------------------------------------------------------------
# PassiveAgent — always end phase, decline effects, cancel chains
# ---------------------------------------------------------------------------

class PassiveAgent:
    """Passive bot: ends phases, declines effects, cancels chains.

    Duels end by deck-out (neither player takes voluntary actions).
    For mandatory selections (e.g. forced position, forced tribute)
    it picks the first available option.
    """

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int:
        pt = prompt.prompt_type

        if pt == PromptType.IDLE_CMD:
            return END_PHASE
        if pt == PromptType.IDLE_SUBMENU:
            return BACK
        if pt == PromptType.BATTLE_MENU:
            return END_PHASE
        if pt == PromptType.BATTLE_SELECT:
            return BACK

        if pt == PromptType.SELECT_CHAIN:
            return CANCEL if prompt.cancelable else 0
        if pt == PromptType.SELECT_EFFECTYN:
            return DECLINE
        if pt == PromptType.SELECT_YESNO:
            return DECLINE

        if pt in (PromptType.SELECT_CARD, PromptType.SELECT_TRIBUTE,
                  PromptType.SELECT_SUM):
            return 0  # translator picks first min_select options

        if pt == PromptType.SELECT_POSITION:
            return 0
        if pt == PromptType.SELECT_PLACE:
            return 0
        if pt == PromptType.SELECT_OPTION:
            return 0

        if pt == PromptType.SELECT_COUNTER:
            return 0  # translator sends all zeros

        if pt == PromptType.SELECT_UNSELECT:
            return FINISH if prompt.finishable else 0

        if pt in (PromptType.ANNOUNCE_RACE, PromptType.ANNOUNCE_ATTRIB):
            return 0
        if pt == PromptType.ANNOUNCE_NUMBER:
            return 0
        if pt == PromptType.ANNOUNCE_CARD:
            return CANCEL
        if pt == PromptType.SORT_CARD:
            return CANCEL

        # UNKNOWN or unhandled
        return CANCEL
