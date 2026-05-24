"""Unit tests for the MUD action translator.

No MUD server required.
"""

from __future__ import annotations

from yugioh_mud.action_translator import ActionTranslator
from yugioh_mud.agent import BACK, END_PHASE
from yugioh_mud.text_parser import ParsedPrompt, PromptType


class TestIdleBattleTranslation:
    """Idle/battle prompts send the option string directly."""

    def test_idle_cmd_sends_option(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.IDLE_CMD, options=["h1", "h3", "b", "e"])
        assert tr.translate(0, prompt) == "h1"
        assert tr.translate(1, prompt) == "h3"
        assert tr.translate(2, prompt) == "b"
        assert tr.translate(3, prompt) == "e"

    def test_idle_cmd_end_phase(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.IDLE_CMD, options=["b", "e"])
        assert tr.translate(END_PHASE, prompt) == "e"

    def test_idle_submenu_sends_letter(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.IDLE_SUBMENU, options=["s", "t", "z"])
        assert tr.translate(0, prompt) == "s"
        assert tr.translate(1, prompt) == "t"
        assert tr.translate(BACK, prompt) == "z"

    def test_battle_menu_sends_option(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.BATTLE_MENU, options=["a", "c", "m", "e"])
        assert tr.translate(0, prompt) == "a"
        assert tr.translate(2, prompt) == "m"

    def test_battle_select_sends_cardspec(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.BATTLE_SELECT, options=["m1", "m2", "z"])
        assert tr.translate(0, prompt) == "m1"
        assert tr.translate(1, prompt) == "m2"

    def test_idle_cmd_fallback_on_empty_options(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.IDLE_CMD, options=[])
        assert tr.translate(0, prompt) == "e"


class TestTranslatorBounds:
    def test_select_card_action_exceeds_options(self):
        """Indices are clamped when action + min_select > len(options)."""
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_CARD,
            options=["Card A", "Card B", "Card C"],
            min_select=2,
            max_select=3,
        )
        # action=2 (0-indexed) → would produce indices 3,4 but only 3 options
        result = tr.translate(2, prompt)
        # Clamped: start = min(2, 3-2) = 1 → indices "2 3"
        assert result == "2 3"

    def test_select_card_action_zero(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_CARD,
            options=["Card A", "Card B", "Card C"],
            min_select=2,
            max_select=3,
        )
        result = tr.translate(0, prompt)
        assert result == "1 2"

    def test_select_tribute_clamped(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.SELECT_TRIBUTE, options=["Mon A", "Mon B"], min_select=2, max_select=2
        )
        # action=1 would overflow — clamped to start=0
        result = tr.translate(1, prompt)
        assert result == "1 2"

    def test_announce_race_clamped(self):
        tr = ActionTranslator()
        prompt = ParsedPrompt(
            PromptType.ANNOUNCE_RACE,
            options=["Warrior", "Spellcaster", "Dragon"],
            min_select=2,
            max_select=2,
        )
        result = tr.translate(3, prompt)
        # Clamped: start = min(3, 3-2) = 1 → "2 3"
        assert result == "2 3"

    def test_sort_card_identity_permutation(self):
        """Translator emits 1-indexed identity for SORT_CARD with any non-negative action."""
        tr = ActionTranslator()
        prompt = ParsedPrompt(PromptType.SORT_CARD, options=[], min_select=3)
        assert tr.translate(0, prompt) == "1 2 3"
