"""Tests for ModelAgent action mapping logic.

The ``map_model_action`` function is tested directly (no torch required).
``ModelAgent`` instantiation tests are guarded behind ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

from dataclasses import dataclass

from yugioh_mud.agent import (
    CANCEL,
    DECLINE,
    END_PHASE,
    FINISH,
    map_model_action,
)
from yugioh_mud.text_parser import ParsedPrompt, PromptType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeStructuredAction:
    """Minimal stand-in for StructuredAction (avoids importing cmd_handler)."""

    category: int = 0
    cardspec: str = ""
    card_code: int = 0
    location: int = 0
    sequence: int = 0
    sub_action: str = ""


def _idle_prompt(*categories: int) -> ParsedPrompt:
    """Build an IDLE_CMD prompt with structured actions of given categories."""
    sa = [FakeStructuredAction(category=c) for c in categories]
    return ParsedPrompt(
        prompt_type=PromptType.IDLE_CMD,
        options=[f"option{i}" for i in range(len(sa))],
        structured_actions=sa,
    )


def _battle_prompt(*categories: int) -> ParsedPrompt:
    """Build a BATTLE_MENU prompt with structured actions of given categories."""
    sa = [FakeStructuredAction(category=c) for c in categories]
    return ParsedPrompt(
        prompt_type=PromptType.BATTLE_MENU,
        options=[f"option{i}" for i in range(len(sa))],
        structured_actions=sa,
    )


def _options_prompt(pt: PromptType, n: int, **kwargs) -> ParsedPrompt:
    return ParsedPrompt(
        prompt_type=pt,
        options=[f"opt{i}" for i in range(n)],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# IDLE_CMD mapping
# ---------------------------------------------------------------------------


class TestIdleCmdMapping:
    def test_index_zero_maps_to_first_action(self):
        prompt = _idle_prompt(0, 5, 1)  # summon, activate, sp_summon
        assert map_model_action(0, prompt) == 0

    def test_index_within_range(self):
        prompt = _idle_prompt(0, 5, 1)
        assert map_model_action(2, prompt) == 2

    def test_index_out_of_range_returns_end_phase(self):
        prompt = _idle_prompt(0, 5)
        assert map_model_action(5, prompt) == END_PHASE

    def test_empty_actions_returns_end_phase(self):
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD)
        assert map_model_action(0, prompt) == END_PHASE

    def test_to_bp_category_returns_index(self):
        prompt = _idle_prompt(0, 5, 6)  # 6 = to_bp
        assert map_model_action(2, prompt) == 2  # handler dispatches via sub_action

    def test_to_ep_category_returns_index(self):
        prompt = _idle_prompt(0, 7)  # 7 = to_ep
        assert map_model_action(1, prompt) == 1  # handler dispatches via sub_action

    def test_all_categories_return_index(self):
        for cat in range(8):
            prompt = _idle_prompt(cat)
            assert map_model_action(0, prompt) == 0


# ---------------------------------------------------------------------------
# BATTLE_MENU mapping
# ---------------------------------------------------------------------------


class TestBattleMenuMapping:
    def test_activate_returns_index(self):
        prompt = _battle_prompt(0, 1)  # activate, attack
        assert map_model_action(0, prompt) == 0

    def test_attack_returns_index(self):
        prompt = _battle_prompt(0, 1)
        assert map_model_action(1, prompt) == 1

    def test_to_m2_returns_index(self):
        prompt = _battle_prompt(0, 1, 2)  # 2 = to_m2
        assert map_model_action(2, prompt) == 2  # handler dispatches via sub_action

    def test_to_ep_returns_index(self):
        prompt = _battle_prompt(0, 3)  # 3 = to_ep
        assert map_model_action(1, prompt) == 1  # handler dispatches via sub_action

    def test_out_of_range_returns_end_phase(self):
        prompt = _battle_prompt(0, 1)
        assert map_model_action(10, prompt) == END_PHASE


# ---------------------------------------------------------------------------
# SELECT_EFFECTYN / SELECT_YESNO mapping
# ---------------------------------------------------------------------------


class TestEffectYNMapping:
    def test_index_zero_accepts(self):
        prompt = _options_prompt(PromptType.SELECT_EFFECTYN, 2)
        assert map_model_action(0, prompt) == 0

    def test_index_one_declines(self):
        prompt = _options_prompt(PromptType.SELECT_EFFECTYN, 2)
        assert map_model_action(1, prompt) == DECLINE

    def test_large_index_declines(self):
        prompt = _options_prompt(PromptType.SELECT_YESNO, 2)
        assert map_model_action(31, prompt) == DECLINE


# ---------------------------------------------------------------------------
# SELECT_CHAIN mapping
# ---------------------------------------------------------------------------


class TestChainMapping:
    def test_valid_index(self):
        prompt = _options_prompt(PromptType.SELECT_CHAIN, 3, cancelable=True)
        assert map_model_action(1, prompt) == 1

    def test_out_of_range_cancels(self):
        prompt = _options_prompt(PromptType.SELECT_CHAIN, 3, cancelable=True)
        assert map_model_action(5, prompt) == CANCEL

    def test_zero_options_cancels(self):
        prompt = _options_prompt(PromptType.SELECT_CHAIN, 0, cancelable=True)
        assert map_model_action(0, prompt) == CANCEL


# ---------------------------------------------------------------------------
# SELECT_UNSELECT mapping
# ---------------------------------------------------------------------------


class TestUnselectMapping:
    def test_valid_index(self):
        prompt = _options_prompt(PromptType.SELECT_UNSELECT, 3, finishable=True)
        assert map_model_action(2, prompt) == 2

    def test_out_of_range_finishes(self):
        prompt = _options_prompt(PromptType.SELECT_UNSELECT, 3, finishable=True)
        assert map_model_action(5, prompt) == FINISH

    def test_zero_options_finishes(self):
        prompt = _options_prompt(PromptType.SELECT_UNSELECT, 0, finishable=True)
        assert map_model_action(0, prompt) == FINISH


# ---------------------------------------------------------------------------
# Generic SELECT_* mapping (clamped)
# ---------------------------------------------------------------------------


class TestGenericSelectMapping:
    def test_select_card_in_range(self):
        prompt = _options_prompt(PromptType.SELECT_CARD, 5)
        assert map_model_action(3, prompt) == 3

    def test_select_card_clamped(self):
        prompt = _options_prompt(PromptType.SELECT_CARD, 5)
        assert map_model_action(10, prompt) == 4  # clamped to last

    def test_select_tribute_clamped(self):
        prompt = _options_prompt(PromptType.SELECT_TRIBUTE, 2)
        assert map_model_action(5, prompt) == 1

    def test_select_position(self):
        prompt = _options_prompt(PromptType.SELECT_POSITION, 3)
        assert map_model_action(1, prompt) == 1

    def test_select_option(self):
        prompt = _options_prompt(PromptType.SELECT_OPTION, 4)
        assert map_model_action(2, prompt) == 2

    def test_zero_options_returns_cancel(self):
        prompt = _options_prompt(PromptType.SELECT_CARD, 0)
        assert map_model_action(0, prompt) == CANCEL


# ---------------------------------------------------------------------------
# Fallback (no game state → PassiveAgent)
# ---------------------------------------------------------------------------


class TestFallbackWithoutGameState:
    """ModelAgent.choose() with game_state=None delegates to PassiveAgent."""

    def test_idle_returns_end_phase(self):
        # map_model_action is the raw mapper; PassiveAgent fallback is tested
        # via ModelAgent.choose() which requires torch. Here we verify the
        # PassiveAgent default directly for the key prompt types.
        from yugioh_mud.agent import PassiveAgent

        agent = PassiveAgent()
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD)
        assert agent.choose(prompt) == END_PHASE

    def test_effectyn_returns_decline(self):
        from yugioh_mud.agent import PassiveAgent

        agent = PassiveAgent()
        prompt = _options_prompt(PromptType.SELECT_EFFECTYN, 2)
        assert agent.choose(prompt) == DECLINE

    def test_chain_returns_cancel(self):
        from yugioh_mud.agent import PassiveAgent

        agent = PassiveAgent()
        prompt = _options_prompt(PromptType.SELECT_CHAIN, 3, cancelable=True)
        assert agent.choose(prompt) == CANCEL
