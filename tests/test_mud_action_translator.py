"""Unit tests for the MUD action translator.

No MUD server required.
"""

from __future__ import annotations

from yugioh_mud.action_translator import ActionTranslator
from yugioh_mud.text_parser import ParsedPrompt, PromptType


class TestTranslatorBounds:
    def test_select_card_action_exceeds_options(self):
        """Indices are clamped when action + min_select > len(options)."""
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_CARD,
            options=["Card A", "Card B", "Card C"],
            min_select=2, max_select=3)
        # action=2 (0-indexed) → would produce indices 3,4 but only 3 options
        result = tr.translate(2, prompt)
        # Clamped: start = min(2, 3-2) = 1 → indices "2 3"
        assert result == "2 3"

    def test_select_card_action_zero(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_CARD,
            options=["Card A", "Card B", "Card C"],
            min_select=2, max_select=3)
        result = tr.translate(0, prompt)
        assert result == "1 2"

    def test_select_tribute_clamped(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_TRIBUTE,
            options=["Mon A", "Mon B"],
            min_select=2, max_select=2)
        # action=1 would overflow — clamped to start=0
        result = tr.translate(1, prompt)
        assert result == "1 2"

    def test_announce_race_clamped(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.ANNOUNCE_RACE,
            options=["Warrior", "Spellcaster", "Dragon"],
            min_select=2, max_select=2)
        result = tr.translate(3, prompt)
        # Clamped: start = min(3, 3-2) = 1 → "2 3"
        assert result == "2 3"
