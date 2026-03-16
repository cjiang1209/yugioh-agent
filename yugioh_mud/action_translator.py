"""Translate agent action ints into MUD text commands.

The ``ActionTranslator`` converts a numeric action (from an ``Agent``)
plus the ``ParsedPrompt`` context into the text string to send to the
MUD server.
"""

from __future__ import annotations

from yugioh_mud.agent import BACK, CANCEL, DECLINE, END_PHASE, FINISH
from yugioh_mud.text_parser import ParsedPrompt, PromptType


class ActionTranslator:
    """Converts ``(action, prompt)`` → MUD text command."""

    def translate(self, action: int, prompt: ParsedPrompt) -> str:
        # -- Meta-commands ---------------------------------------------------
        if action == END_PHASE:
            return "e"
        if action == CANCEL:
            return "c"
        if action == DECLINE:
            return "n"
        if action == BACK:
            return "z"
        if action == FINISH:
            return "f"

        # -- Index-based responses -------------------------------------------
        pt = prompt.prompt_type

        # DuelMenu prompts: 1-indexed number
        if pt in (PromptType.SELECT_POSITION, PromptType.SELECT_OPTION):
            return str(action + 1)

        # Numbered card selection: space-separated 1-indexed numbers
        # Pick first min_select items starting from the chosen index,
        # clamped so indices don't exceed the number of available options.
        if pt in (PromptType.SELECT_CARD, PromptType.SELECT_TRIBUTE,
                  PromptType.SELECT_SUM):
            n = len(prompt.options) if prompt.options else prompt.min_select
            start = min(action, max(n - prompt.min_select, 0))
            indices = list(range(start + 1, start + 1 + prompt.min_select))
            return " ".join(str(i) for i in indices)

        # Chain selection: send the spec string from options
        if pt == PromptType.SELECT_CHAIN:
            if action < len(prompt.options):
                return prompt.options[action]
            return prompt.options[0] if prompt.options else "1"

        # Place selection: send the spec string(s) from options
        if pt == PromptType.SELECT_PLACE:
            specs = prompt.options[action:action + prompt.min_select]
            return " ".join(specs) if specs else prompt.options[0]

        # Counter: send N zeros (one per card)
        if pt == PromptType.SELECT_COUNTER:
            return " ".join("0" for _ in range(prompt.min_select))

        # Unselect: send 1-indexed card number
        if pt == PromptType.SELECT_UNSELECT:
            return str(action + 1)

        # Announce race / attrib: 1-indexed number(s)
        if pt in (PromptType.ANNOUNCE_RACE, PromptType.ANNOUNCE_ATTRIB):
            n = len(prompt.options) if prompt.options else prompt.min_select
            start = min(action, max(n - prompt.min_select, 0))
            indices = list(range(start + 1, start + 1 + prompt.min_select))
            return " ".join(str(i) for i in indices)

        # Announce number: the literal number from options
        if pt == PromptType.ANNOUNCE_NUMBER:
            if action < len(prompt.options):
                return prompt.options[action]
            return prompt.options[0] if prompt.options else "1"

        # Sort card: space-separated 1-indexed order (identity permutation)
        if pt == PromptType.SORT_CARD:
            return " ".join(str(i + 1) for i in range(prompt.min_select))

        # Announce card: no good default, send cancel
        if pt == PromptType.ANNOUNCE_CARD:
            return "c"

        # Fallback
        return str(action + 1)
