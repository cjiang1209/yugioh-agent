"""Unit tests for the MUD duel text parser.

Uses exact server output patterns derived from the yugioh-game source code.
No MUD server required.
"""

from __future__ import annotations

import pytest

from yugioh_mud.text_parser import (
    EventType,
    MUDTextParser,
    ParsedEvent,
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

    def test_idle_cmd_extracts_letter_options(self, parser: MUDTextParser):
        """IDLE_CMD extracts b and e letter commands into options."""
        parser.feed_line("Select a card on which to perform an action.")
        parser.feed_line(
            "h shows your hand, tab and tab2 shows your or the "
            "opponent's table, ? shows usable cards.")
        parser.feed_line("b: Enter the battle phase.")
        parser.feed_line("e: End phase.")
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_CMD
        assert "b" in prompt.options
        assert "e" in prompt.options

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

    def test_idle_cmd_no_bp_only_e(self, parser: MUDTextParser):
        """When no battle phase, options has only 'e'."""
        parser.feed_line("Select a card on which to perform an action.")
        parser.feed_line("e: End phase.")
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.options == ["e"]

    def test_idle_submenu(self, parser: MUDTextParser):
        prompt = parser.feed_line("Select action for Dark Magician")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_SUBMENU

    def test_idle_submenu_accumulates_actions(self, parser: MUDTextParser):
        """IDLE_SUBMENU accumulates letter options when in idle context."""
        parser.feed_line("Select a card on which to perform an action.")
        parser.feed_line("b: Enter the battle phase.")
        parser.feed_line("e: End phase.")
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None  # consume IDLE_CMD

        # Now simulate selecting a card — submenu letters arrive
        assert parser.feed_line("s: Summon.") is None
        assert parser.feed_line("t: Set.") is None
        assert parser.feed_line("z: back.") is None
        prompt = parser.feed_line("Select action for Dark Magician")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_SUBMENU
        assert "s" in prompt.options
        assert "t" in prompt.options
        assert "z" in prompt.options

    def test_idle_submenu_multi_effect(self, parser: MUDTextParser):
        """IDLE_SUBMENU handles multi-effect va, vb options."""
        parser.feed_line("Select a card on which to perform an action.")
        parser.feed_line("e: End phase.")
        parser.feed_line("Select a card:")  # consume IDLE_CMD

        assert parser.feed_line("va: Activate effect 1.") is None
        assert parser.feed_line("vb: Activate effect 2.") is None
        assert parser.feed_line("z: back.") is None
        prompt = parser.feed_line("Select action for Blue-Eyes White Dragon")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.IDLE_SUBMENU
        assert "va" in prompt.options
        assert "vb" in prompt.options
        assert "z" in prompt.options


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

    def test_battle_menu_extracts_options(self, parser: MUDTextParser):
        """BATTLE_MENU extracts letter commands into options."""
        parser.feed_line("Battle menu:")
        parser.feed_line("a: Attack.")
        parser.feed_line("c: activate.")
        parser.feed_line("m: Main phase 2.")
        parser.feed_line("e: End phase.")
        prompt = parser.feed_line("Select an option:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_MENU
        assert prompt.options == ["a", "c", "m", "e"]

    def test_battle_menu_only_end(self, parser: MUDTextParser):
        assert parser.feed_line("Battle menu:") is None
        assert parser.feed_line("e: End phase.") is None
        prompt = parser.feed_line("Select an option:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_MENU

    def test_battle_menu_only_end_options(self, parser: MUDTextParser):
        parser.feed_line("Battle menu:")
        parser.feed_line("e: End phase.")
        prompt = parser.feed_line("Select an option:")
        assert prompt is not None
        assert prompt.options == ["e"]

    def test_battle_attack_submenu(self, parser: MUDTextParser):
        assert parser.feed_line("Select card to attack with:") is None
        assert parser.feed_line("m1: Dark Magician (2500/2100)") is None
        assert parser.feed_line("z: back.") is None
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_SELECT

    def test_battle_attack_submenu_extracts_cardspecs(self, parser: MUDTextParser):
        """BATTLE_SELECT extracts cardspec and z options."""
        parser.feed_line("Select card to attack with:")
        parser.feed_line("m1: Dark Magician (2500/2100)")
        parser.feed_line("m2: Blue-Eyes White Dragon (3000/2500)")
        parser.feed_line("z: back.")
        prompt = parser.feed_line("Select a card:")
        assert prompt is not None
        assert prompt.prompt_type == PromptType.BATTLE_SELECT
        assert "m1" in prompt.options
        assert "m2" in prompt.options
        assert "z" in prompt.options

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
# Event parsing (informational lines → ParsedEvent)
# ---------------------------------------------------------------------------

class TestEventTurn:
    def test_your_turn(self, parser: MUDTextParser):
        ev = parser.feed_line("Your turn.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.NEW_TURN
        assert ev.player == "you"
        assert ev.is_opponent is False

    def test_opponent_turn(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2's turn.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.NEW_TURN
        assert ev.player == "Player2"
        assert ev.is_opponent is True


class TestEventPhase:
    @pytest.mark.parametrize("phase_str", [
        "draw phase", "standby phase", "main1 phase",
        "battle start phase", "battle phase", "main2 phase",
        "end phase",
    ])
    def test_phase(self, parser: MUDTextParser, phase_str: str):
        ev = parser.feed_line(f"entering {phase_str}.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.NEW_PHASE
        assert ev.phase == phase_str


class TestEventLP:
    def test_your_damage(self, parser: MUDTextParser):
        ev = parser.feed_line("Your lp decreased by 1500, now 6500")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DAMAGE
        assert ev.player == "you"
        assert ev.amount == 1500
        assert ev.new_lp == 6500

    def test_opp_damage(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2's lp decreased by 2000, now 6000")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DAMAGE
        assert ev.player == "Player2"
        assert ev.is_opponent is True
        assert ev.amount == 2000
        assert ev.new_lp == 6000

    def test_your_recover(self, parser: MUDTextParser):
        ev = parser.feed_line("Your lp increased by 500, now 8500")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.RECOVER
        assert ev.amount == 500
        assert ev.new_lp == 8500

    def test_opp_recover(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2's lp increased by 500, now 8500")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.RECOVER
        assert ev.is_opponent is True

    def test_your_pay_lp(self, parser: MUDTextParser):
        ev = parser.feed_line("You pay 1000 LP. Your LP is now 7000.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.PAY_LP
        assert ev.amount == 1000
        assert ev.new_lp == 7000

    def test_opp_pay_lp(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 pays 800 LP. Their LP is now 7200.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.PAY_LP
        assert ev.player == "Player2"
        assert ev.is_opponent is True
        assert ev.amount == 800
        assert ev.new_lp == 7200


class TestEventDraw:
    def test_your_draw(self, parser: MUDTextParser):
        ev = parser.feed_line("Drew 1 cards:")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DRAW
        assert ev.player == "you"
        assert ev.amount == 1

    def test_opp_draw(self, parser: MUDTextParser):
        ev = parser.feed_line("Opponent drew 1 cards.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DRAW
        assert ev.is_opponent is True
        assert ev.amount == 1

    def test_opp_draw_with_name(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 drew 2 cards.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DRAW
        assert ev.player == "Player2"
        assert ev.is_opponent is True
        assert ev.amount == 2


class TestEventSummon:
    def test_normal_summon(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player1 summoning Dark Magician (2500/2100) "
            "in face-up attack position.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SUMMON
        assert ev.player == "Player1"
        assert ev.card_name == "Dark Magician"
        assert ev.position == "face-up attack"

    def test_special_summon(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player1 special summoning Blue-Eyes White Dragon (3000/2500) "
            "in face-up attack position.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SP_SUMMON
        assert ev.card_name == "Blue-Eyes White Dragon"

    def test_special_summon_link(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player1 special summoning Decode Talker (2300) "
            "in face-up attack position.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SP_SUMMON
        assert ev.card_name == "Decode Talker"

    def test_flip_summon(self, parser: MUDTextParser):
        ev = parser.feed_line("Player1 flip summons Man-Eater Bug (m2).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.FLIP_SUMMON
        assert ev.card_name == "Man-Eater Bug"
        assert ev.card_spec == "m2"

    def test_set_self(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "You set m1 (Kuriboh) in face-down defense position.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SET
        assert ev.player == "you"
        assert ev.card_spec == "m1"
        assert ev.card_name == "Kuriboh"
        assert ev.position == "face-down defense"

    def test_set_opp(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2 sets m1 in face-down defense position.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SET
        assert ev.is_opponent is True
        assert ev.card_spec == "m1"


class TestEventPosChange:
    def test_pos_change(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "The position of card m1 (Dark Magician) "
            "was changed to face-up defense.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.POS_CHANGE
        assert ev.card_spec == "m1"
        assert ev.card_name == "Dark Magician"
        assert ev.position == "face-up defense"


class TestEventAttack:
    def test_attack_targeted(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player1 prepares to attack om1 (Kuriboh) "
            "with m1 (Dark Magician)")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.ATTACK
        assert ev.player == "Player1"
        assert ev.card_spec == "m1"
        assert ev.card_name == "Dark Magician"
        assert ev.target_spec == "om1"
        assert ev.target_name == "Kuriboh"

    def test_attack_direct(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player1 prepares to attack with m1 (Dark Magician)")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.ATTACK
        assert ev.card_spec == "m1"
        assert ev.card_name == "Dark Magician"
        assert ev.target_spec == ""


class TestEventChaining:
    def test_your_chaining(self, parser: MUDTextParser):
        ev = parser.feed_line("Activating s1 (Mirror Force)")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.CHAINING
        assert ev.player == "you"
        assert ev.card_spec == "s1"
        assert ev.card_name == "Mirror Force"

    def test_opp_chaining(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 activating s1 (Trap Hole)")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.CHAINING
        assert ev.is_opponent is True
        assert ev.card_name == "Trap Hole"


class TestEventMovement:
    def test_destroy(self, parser: MUDTextParser):
        ev = parser.feed_line("Card m1 (Dark Magician) destroyed.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DESTROY
        assert ev.card_spec == "m1"
        assert ev.card_name == "Dark Magician"

    def test_your_to_gy(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "your card m1 (Dark Magician) was sent to the graveyard.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_GRAVEYARD
        assert ev.player == "you"
        assert ev.card_name == "Dark Magician"

    def test_opp_to_gy(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2's card m1 (Kuriboh) was sent to the graveyard.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_GRAVEYARD
        assert ev.is_opponent is True

    def test_your_banished(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "your card m1 (Dark Magician) was banished.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.BANISHED

    def test_opp_banished(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2's card m1 (Kuriboh) was banished.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.BANISHED
        assert ev.is_opponent is True

    def test_your_to_hand(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Card m1 (Dark Magician) returned to hand.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_HAND
        assert ev.player == "you"

    def test_opp_to_hand(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2's card m1 (Kuriboh) returned to their hand.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_HAND
        assert ev.is_opponent is True

    def test_your_to_deck(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "your card m1 (Dark Magician) returned to your deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_DECK

    def test_opp_to_deck(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2's card m1 (Kuriboh) returned to their deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_DECK
        assert ev.is_opponent is True

    def test_your_to_extra_deck(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "your card m1 (Decode Talker) returned to your extra deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_EXTRA_DECK

    def test_opp_to_extra_deck(self, parser: MUDTextParser):
        ev = parser.feed_line(
            "Player2's card m1 (Decode Talker) "
            "returned to their extra deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TO_EXTRA_DECK
        assert ev.is_opponent is True

    def test_your_tribute(self, parser: MUDTextParser):
        ev = parser.feed_line("You tribute m1 (Kuriboh).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TRIBUTE
        assert ev.player == "you"

    def test_opp_tribute(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 tributes m1 (Kuriboh).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.TRIBUTE
        assert ev.is_opponent is True

    def test_your_discard(self, parser: MUDTextParser):
        ev = parser.feed_line("you discarded h1 (Kuriboh).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DISCARD
        assert ev.player == "you"

    def test_opp_discard(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 discarded h1 (Kuriboh).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.DISCARD
        assert ev.is_opponent is True


class TestEventMisc:
    def test_equip(self, parser: MUDTextParser):
        ev = parser.feed_line("Axe of Despair equipped to Dark Magician.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.EQUIP
        assert ev.card_name == "Axe of Despair"
        assert ev.target_name == "Dark Magician"

    def test_your_shuffle(self, parser: MUDTextParser):
        ev = parser.feed_line("you shuffled your deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SHUFFLE
        assert ev.player == "you"

    def test_opp_shuffle(self, parser: MUDTextParser):
        ev = parser.feed_line("Player2 shuffled their deck.")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.SHUFFLE
        assert ev.is_opponent is True


class TestEventWinLose:
    def test_win(self, parser: MUDTextParser):
        ev = parser.feed_line("You won (LP became 0).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.WIN

    def test_lose(self, parser: MUDTextParser):
        ev = parser.feed_line("You lost (ran out of cards to draw).")
        assert isinstance(ev, ParsedEvent)
        assert ev.event_type == EventType.LOSE


# ---------------------------------------------------------------------------
# Unrecognised lines — should still return None
# ---------------------------------------------------------------------------

class TestUnrecognised:
    @pytest.mark.parametrize("line", [
        "",
        "Waiting for opponent.",
        "begin damage",
        "end damage",
    ])
    def test_unrecognised_returns_none(self, parser: MUDTextParser, line: str):
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
