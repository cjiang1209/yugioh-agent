"""Unit tests for the MUD duel text parser.

Uses exact server output patterns derived from the yugioh-game source code.
No MUD server required.
"""

from __future__ import annotations

import pytest

from yugioh_mud.text_parser import (
    MUDTextParser,
    ParsedPrompt,
    PromptType,
    is_duel_end,
)


@pytest.fixture
def parser() -> MUDTextParser:
    return MUDTextParser()


# ---------------------------------------------------------------------------
# Idle phase
# ---------------------------------------------------------------------------

class TestIdleCmd:
    def test_idle_cmd_basic(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select a card on which to perform an action.") is None
        assert parser.feed_line(
            "h shows your hand, tab and tab2 shows your or the "
            "opponent's table, ? shows usable cards.") is None
        assert parser.feed_line("b: Enter the battle phase.") is None
        assert parser.feed_line("e: End phase.") is None
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_CMD

    def test_idle_cmd_no_battle_phase(self, parser: MUDTextParser):
        """When to_bp is false, only 'e:' is present."""
        assert parser.feed_line(
            "Select a card on which to perform an action.") is None
        assert parser.feed_line(
            "h shows your hand, tab and tab2 shows your or the "
            "opponent's table, ? shows usable cards.") is None
        assert parser.feed_line("e: End phase.") is None
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_CMD

    def test_idle_submenu(self, parser: MUDTextParser):
        prompt = parser.feed_line("Select action for Dark Magician")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_SUBMENU


# ---------------------------------------------------------------------------
# Battle menu
# ---------------------------------------------------------------------------

class TestBattleMenu:
    def test_battle_menu(self, parser: MUDTextParser):
        assert parser.feed_line("Battle menu:") is None
        assert parser.feed_line("a: Attack.") is None
        assert parser.feed_line("c: activate.") is None
        assert parser.feed_line("m: Main phase 2.") is None
        assert parser.feed_line("e: End phase.") is None
        prompt = parser.feed_line("Select an option:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_MENU

    def test_battle_menu_only_end(self, parser: MUDTextParser):
        assert parser.feed_line("Battle menu:") is None
        assert parser.feed_line("e: End phase.") is None
        prompt = parser.feed_line("Select an option:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_MENU

    def test_battle_attack_submenu(self, parser: MUDTextParser):
        assert parser.feed_line("Select card to attack with:") is None
        assert parser.feed_line("m1: Dark Magician (2500/2100)") is None
        assert parser.feed_line("z: back.") is None
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_SELECT

    def test_battle_activate_submenu(self, parser: MUDTextParser):
        assert parser.feed_line("Select card to activate:") is None
        assert parser.feed_line("s1: Mirror Force (0/0)") is None
        assert parser.feed_line("z: back.") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_SELECT


# ---------------------------------------------------------------------------
# Select card / tribute
# ---------------------------------------------------------------------------

class TestSelectCard:
    def test_select_card(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select 1 to 3 cards separated by spaces:") is None
        assert parser.feed_line("1: Dark Magician (2500/2100)") is None
        assert parser.feed_line("2: Blue-Eyes White Dragon (3000/2500)") is None
        assert parser.feed_line("3: Red-Eyes Black Dragon (2400/2000)") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_CARD
        assert prompt.min_select == 1
        assert prompt.max_select == 3
        assert len(prompt.options) == 3
        assert "Dark Magician (2500/2100)" in prompt.options[0]

    def test_select_tribute(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select 1 to 2 cards to tribute separated by spaces:") is None
        assert parser.feed_line("1: Kuriboh") is None
        assert parser.feed_line("2: Sangan") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_TRIBUTE
        assert prompt.min_select == 1
        assert prompt.max_select == 2
        assert len(prompt.options) == 2


# ---------------------------------------------------------------------------
# Select chain
# ---------------------------------------------------------------------------

class TestSelectChain:
    def test_chain_optional(self, parser: MUDTextParser):
        assert parser.feed_line("Select chain (c to cancel):") is None
        assert parser.feed_line("s1: Mirror Force") is None
        prompt = parser.feed_line("Select card to chain (c = cancel):")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_CHAIN
        assert prompt.cancelable is True
        assert "s1" in prompt.options

    def test_chain_forced(self, parser: MUDTextParser):
        assert parser.feed_line("Select chain:") is None
        assert parser.feed_line("m1a: Trap Hole") is None
        prompt = parser.feed_line("Select card to chain:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_CHAIN
        assert prompt.cancelable is False
        assert "m1a" in prompt.options

    def test_chain_with_effect_desc(self, parser: MUDTextParser):
        assert parser.feed_line("Select chain (c to cancel):") is None
        assert parser.feed_line(
            "m1a (Dark Magician): Destroy 1 card") is None
        prompt = parser.feed_line("Select card to chain (c = cancel):")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_CHAIN
        assert "m1a" in prompt.options


# ---------------------------------------------------------------------------
# Select effectyn / yesno
# ---------------------------------------------------------------------------

class TestEffectYN:
    def test_effectyn(self, parser: MUDTextParser):
        prompt = parser.feed_line(
            "Do you want to use the effect from Mirror Force in s1?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_EFFECTYN

    def test_effectyn_with_description(self, parser: MUDTextParser):
        # Multi-line frame with embedded newline
        prompt = parser.feed_line(
            "Do you want to use the effect from Mirror Force in s1?\n"
            "Destroy all attacking monsters.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_EFFECTYN


# ---------------------------------------------------------------------------
# Yes/No (MSG_YESNO — distinct from SELECT_EFFECTYN)
# ---------------------------------------------------------------------------

class TestYesNo:
    def test_replay_battle(self, parser: MUDTextParser):
        prompt = parser.feed_line(
            "Replay, do you want to continue the Battle?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_YESNO

    def test_direct_attack(self, parser: MUDTextParser):
        prompt = parser.feed_line("Do you want to Attack Directly?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_YESNO

    def test_normal_summon_without_tribute(self, parser: MUDTextParser):
        prompt = parser.feed_line("Normal Summon without tribute(s)?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_YESNO

    def test_chain_another_card(self, parser: MUDTextParser):
        prompt = parser.feed_line("Chain another card?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_YESNO

    def test_effectyn_takes_priority(self, parser: MUDTextParser):
        """SELECT_EFFECTYN pattern matches before the generic YESNO fallback."""
        prompt = parser.feed_line(
            "Do you want to use the effect from Mirror Force in s1?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_EFFECTYN


# ---------------------------------------------------------------------------
# Select position
# ---------------------------------------------------------------------------

class TestSelectPosition:
    def test_position(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select position for Dark Magician:") is None
        assert parser.feed_line("[1] Face-up attack") is None
        assert parser.feed_line("[2] Face-down defense") is None
        prompt = parser.feed_line("Type a number or @abort to abort.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_POSITION
        assert len(prompt.options) == 2
        assert prompt.options[0] == "Face-up attack"
        assert prompt.options[1] == "Face-down defense"


# ---------------------------------------------------------------------------
# Select place
# ---------------------------------------------------------------------------

class TestSelectPlace:
    def test_place_single(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select place for card, one of m1, m2, m3.") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_PLACE
        assert prompt.options == ["m1", "m2", "m3"]
        assert prompt.min_select == 1

    def test_place_multi(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select 2 places for card, from m1, m2, s1, s2.") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_PLACE
        assert prompt.min_select == 2
        assert prompt.max_select == 2
        assert len(prompt.options) == 4


# ---------------------------------------------------------------------------
# Select option
# ---------------------------------------------------------------------------

class TestSelectOption:
    def test_select_option(self, parser: MUDTextParser):
        # DuelMenu with title="Select option:" and prompt="Select option:"
        assert parser.feed_line("Select option:") is None
        assert parser.feed_line("[1] Draw 1 card") is None
        assert parser.feed_line("[2] Destroy a card") is None
        prompt = parser.feed_line("Select option:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_OPTION
        assert len(prompt.options) == 2
        assert prompt.options[0] == "Draw 1 card"


# ---------------------------------------------------------------------------
# Select sum
# ---------------------------------------------------------------------------

class TestSelectSum:
    def test_select_sum(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Select cards with a total value of 8, "
            "seperated by spaces.") is None
        assert parser.feed_line(
            "Mystical Elf must be selected, "
            "automatically selected.") is None
        assert parser.feed_line("1: Kuriboh (1 or 2)") is None
        assert parser.feed_line("2: Sangan (3)") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_SUM
        assert len(prompt.options) == 2


# ---------------------------------------------------------------------------
# Select counter
# ---------------------------------------------------------------------------

class TestSelectCounter:
    def test_select_counter(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Type new Spell Counter for 2 cards, "
            "separated by spaces.") is None
        assert parser.feed_line("Dark Magician (3)") is None
        assert parser.feed_line("Breaker the Magical Warrior (1)") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_COUNTER
        assert prompt.min_select == 2
        assert prompt.max_select == 2


# ---------------------------------------------------------------------------
# Select unselect card
# ---------------------------------------------------------------------------

class TestSelectUnselect:
    def test_unselect_finishable(self, parser: MUDTextParser):
        # Multi-line frame with embedded newline
        assert parser.feed_line(
            "Check or uncheck 1 to 3 cards by entering their number\n"
            "Enter f to finish") is None
        assert parser.feed_line("1: Dark Magician (unchecked)") is None
        assert parser.feed_line("2: Blue-Eyes (checked)") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_UNSELECT
        assert prompt.min_select == 1
        assert prompt.max_select == 3
        assert prompt.finishable is True

    def test_unselect_cancelable(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Check or uncheck 1 to 2 cards by entering their number\n"
            "Enter c to cancel") is None
        assert parser.feed_line("1: Kuriboh (unchecked)") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_UNSELECT
        assert prompt.cancelable is True
        assert prompt.finishable is False


# ---------------------------------------------------------------------------
# Announce race / attrib / number / card
# ---------------------------------------------------------------------------

class TestAnnounce:
    def test_announce_race(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Type 1 race separated by spaces.") is None
        assert parser.feed_line("1: Warrior") is None
        assert parser.feed_line("2: Spellcaster") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.ANNOUNCE_RACE
        assert prompt.min_select == 1
        assert len(prompt.options) == 2

    def test_announce_attrib(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Type 1 attribute separated by spaces.") is None
        assert parser.feed_line("1. Dark") is None
        assert parser.feed_line("2. Light") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.ANNOUNCE_ATTRIB
        assert prompt.min_select == 1

    def test_announce_number(self, parser: MUDTextParser):
        prompt = parser.feed_line(
            "Select a number, one of: 4, 6, 7, 8, 12")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.ANNOUNCE_NUMBER
        assert prompt.options == ["4", "6", "7", "8", "12"]

    def test_announce_card(self, parser: MUDTextParser):
        assert parser.feed_line("Enter the name of a card:") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.ANNOUNCE_CARD


# ---------------------------------------------------------------------------
# Sort card
# ---------------------------------------------------------------------------

class TestSortCard:
    def test_sort_card(self, parser: MUDTextParser):
        assert parser.feed_line(
            "Sort 3 cards by entering numbers separated by spaces "
            "(c = cancel):") is None
        assert parser.feed_line("1: Dark Magician") is None
        assert parser.feed_line("2: Blue-Eyes White Dragon") is None
        assert parser.feed_line("3: Red-Eyes Black Dragon") is None
        prompt = parser.feed_line("Enter a line of text.")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SORT_CARD
        assert prompt.min_select == 3
        assert prompt.cancelable is True
        assert len(prompt.options) == 3


# ---------------------------------------------------------------------------
# Informational lines — should return None
# ---------------------------------------------------------------------------

class TestInformational:
    @pytest.mark.parametrize("line", [
        "Your turn.",
        "entering main1 phase.",
        "entering battle phase.",
        "Player2 drew.",
        "Player1 normal summoned Dark Magician.",
        "Player1's LP: 8000 -> 7500",
        "",
        "Waiting for opponent.",
    ])
    def test_info_lines_return_none(self, parser: MUDTextParser, line: str):
        assert parser.feed_line(line) is None


# ---------------------------------------------------------------------------
# Duel end detection
# ---------------------------------------------------------------------------

class TestDuelEnd:
    @pytest.mark.parametrize("line,expected", [
        ("You won (LP became 0).", True),
        ("You lost (LP became 0).", True),
        ("You won (ran out of cards to draw).", True),
        ("You lost (ran out of cards to draw).", True),
        ("You scooped.", True),
        ("The duel was cancelled.", True),
        ("Your turn.", False),
        ("entering main1 phase.", False),
        ("", False),
    ])
    def test_is_duel_end(self, line: str, expected: bool):
        assert is_duel_end(line) is expected


# ---------------------------------------------------------------------------
# Parser reset / multi-prompt sequence
# ---------------------------------------------------------------------------

class TestMultiPrompt:
    def test_consecutive_prompts(self, parser: MUDTextParser):
        """Parser correctly handles multiple prompts in sequence."""
        # First prompt: idle cmd
        parser.feed_line("Select a card on which to perform an action.")
        parser.feed_line("e: End phase.")
        p1 = parser.feed_line("Select a card:")
        assert p1 is not None
        assert p1.prompt_type == PromptType.IDLE_CMD

        # Second prompt: effectyn
        p2 = parser.feed_line(
            "Do you want to use the effect from Trap Hole in s1?")
        assert p2 is not None
        assert p2.prompt_type == PromptType.SELECT_EFFECTYN

        # Third prompt: chain
        parser.feed_line("Select chain (c to cancel):")
        parser.feed_line("s1: Mirror Force")
        p3 = parser.feed_line("Select card to chain (c = cancel):")
        assert p3 is not None
        assert p3.prompt_type == PromptType.SELECT_CHAIN

    def test_reset(self, parser: MUDTextParser):
        """reset() clears accumulation state."""
        parser.feed_line("Select a card on which to perform an action.")
        parser.reset()
        # After reset, should be back in scanning mode
        prompt = parser.feed_line(
            "Do you want to use the effect from Mirror Force in s1?")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.SELECT_EFFECTYN
