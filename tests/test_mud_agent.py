"""Tests for MUD bot agents (RandomAgent)."""

from __future__ import annotations

import pytest

from yugioh_mud.agent import (
    BACK,
    CANCEL,
    DECLINE,
    END_PHASE,
    FINISH,
    PassiveAgent,
    RandomAgent,
)
from yugioh_mud.text_parser import ParsedPrompt, PromptType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt(
    ptype: PromptType,
    n_options: int = 3,
    cancelable: bool = False,
    finishable: bool = False,
    min_select: int = 1,
    max_select: int = 1,
) -> ParsedPrompt:
    options = [f"opt{i}" for i in range(n_options)]
    return ParsedPrompt(
        prompt_type=ptype,
        options=options,
        min_select=min_select,
        max_select=max_select,
        cancelable=cancelable,
        finishable=finishable,
    )


# ---------------------------------------------------------------------------
# RandomAgent: valid actions per prompt type
# ---------------------------------------------------------------------------

class TestRandomAgentValidActions:
    """RandomAgent returns valid actions for every prompt type."""

    def test_idle_cmd(self):
        agent = RandomAgent(seed=0)
        p = _prompt(PromptType.IDLE_CMD, n_options=4)
        for _ in range(50):
            a = agent.choose(p)
            assert a == END_PHASE or 0 <= a < 4

    def test_battle_menu(self):
        agent = RandomAgent(seed=1)
        p = _prompt(PromptType.BATTLE_MENU, n_options=3)
        for _ in range(50):
            a = agent.choose(p)
            assert a == END_PHASE or 0 <= a < 3

    def test_idle_submenu(self):
        agent = RandomAgent(seed=2)
        p = _prompt(PromptType.IDLE_SUBMENU, n_options=2)
        for _ in range(50):
            a = agent.choose(p)
            assert a == BACK or 0 <= a < 2

    def test_battle_select(self):
        agent = RandomAgent(seed=3)
        p = _prompt(PromptType.BATTLE_SELECT, n_options=2)
        for _ in range(50):
            a = agent.choose(p)
            assert a == BACK or 0 <= a < 2

    def test_select_chain_cancelable(self):
        agent = RandomAgent(seed=4)
        p = _prompt(PromptType.SELECT_CHAIN, n_options=2, cancelable=True)
        for _ in range(50):
            a = agent.choose(p)
            assert a == CANCEL or 0 <= a < 2

    def test_select_chain_not_cancelable(self):
        agent = RandomAgent(seed=5)
        p = _prompt(PromptType.SELECT_CHAIN, n_options=2, cancelable=False)
        for _ in range(50):
            a = agent.choose(p)
            assert 0 <= a < 2

    def test_select_effectyn(self):
        agent = RandomAgent(seed=6)
        p = _prompt(PromptType.SELECT_EFFECTYN, n_options=0)
        for _ in range(50):
            a = agent.choose(p)
            assert a in (0, DECLINE)

    def test_select_yesno(self):
        agent = RandomAgent(seed=7)
        p = _prompt(PromptType.SELECT_YESNO, n_options=0)
        for _ in range(50):
            a = agent.choose(p)
            assert a in (0, DECLINE)

    def test_select_unselect_finishable(self):
        agent = RandomAgent(seed=8)
        p = _prompt(PromptType.SELECT_UNSELECT, n_options=3, finishable=True)
        for _ in range(50):
            a = agent.choose(p)
            assert a == FINISH or 0 <= a < 3

    def test_select_unselect_not_finishable(self):
        agent = RandomAgent(seed=9)
        p = _prompt(PromptType.SELECT_UNSELECT, n_options=3, finishable=False)
        for _ in range(50):
            a = agent.choose(p)
            assert 0 <= a < 3

    @pytest.mark.parametrize("ptype", [
        PromptType.SELECT_CARD,
        PromptType.SELECT_TRIBUTE,
        PromptType.SELECT_POSITION,
        PromptType.SELECT_PLACE,
        PromptType.SELECT_OPTION,
        PromptType.SELECT_SUM,
        PromptType.SELECT_COUNTER,
        PromptType.ANNOUNCE_RACE,
        PromptType.ANNOUNCE_ATTRIB,
        PromptType.ANNOUNCE_NUMBER,
    ])
    def test_standard_selections(self, ptype):
        agent = RandomAgent(seed=10)
        p = _prompt(ptype, n_options=5)
        for _ in range(50):
            a = agent.choose(p)
            assert 0 <= a < 5

    def test_sort_card(self):
        agent = RandomAgent(seed=11)
        p = _prompt(PromptType.SORT_CARD, n_options=3)
        assert agent.choose(p) == CANCEL

    def test_announce_card(self):
        agent = RandomAgent(seed=12)
        p = _prompt(PromptType.ANNOUNCE_CARD, n_options=0)
        assert agent.choose(p) == CANCEL

    def test_unknown(self):
        agent = RandomAgent(seed=13)
        p = _prompt(PromptType.UNKNOWN, n_options=0)
        assert agent.choose(p) == CANCEL


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestRandomAgentReproducibility:

    def test_same_seed_same_sequence(self):
        """Two agents with the same seed produce identical action sequences."""
        p = _prompt(PromptType.IDLE_CMD, n_options=5)
        a1 = RandomAgent(seed=42)
        a2 = RandomAgent(seed=42)
        seq1 = [a1.choose(p) for _ in range(100)]
        seq2 = [a2.choose(p) for _ in range(100)]
        assert seq1 == seq2

    def test_different_seed_different_sequence(self):
        """Different seeds produce different sequences (with high probability)."""
        p = _prompt(PromptType.IDLE_CMD, n_options=5)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        seq1 = [a1.choose(p) for _ in range(100)]
        seq2 = [a2.choose(p) for _ in range(100)]
        assert seq1 != seq2

    def test_no_seed_nondeterministic(self):
        """Without seed, agent should still return valid actions."""
        agent = RandomAgent(seed=None)
        p = _prompt(PromptType.IDLE_CMD, n_options=3)
        a = agent.choose(p)
        assert a == END_PHASE or 0 <= a < 3


# ---------------------------------------------------------------------------
# All prompt types handled
# ---------------------------------------------------------------------------

class TestRandomAgentCompleteness:

    def test_all_prompt_types_handled(self):
        """RandomAgent does not raise for any PromptType."""
        agent = RandomAgent(seed=99)
        for pt in PromptType:
            p = _prompt(pt, n_options=3, cancelable=True, finishable=True)
            a = agent.choose(p)
            # Must return an int
            assert isinstance(a, int)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRandomAgentEdgeCases:

    def test_zero_options_select_card(self):
        """Empty options list defaults to 0."""
        agent = RandomAgent(seed=0)
        p = _prompt(PromptType.SELECT_CARD, n_options=0)
        assert agent.choose(p) == 0

    def test_zero_options_chain_not_cancelable(self):
        """Empty chain with no cancel falls back to CANCEL."""
        agent = RandomAgent(seed=0)
        p = _prompt(PromptType.SELECT_CHAIN, n_options=0, cancelable=False)
        assert agent.choose(p) == CANCEL

    def test_zero_options_unselect_not_finishable(self):
        """Empty unselect with no finish falls back to FINISH."""
        agent = RandomAgent(seed=0)
        p = _prompt(PromptType.SELECT_UNSELECT, n_options=0, finishable=False)
        assert agent.choose(p) == FINISH

    def test_menu_returns_end_phase_sometimes(self):
        """Over many trials, END_PHASE should appear for menu prompts."""
        agent = RandomAgent(seed=42)
        p = _prompt(PromptType.IDLE_CMD, n_options=3)
        actions = {agent.choose(p) for _ in range(200)}
        assert END_PHASE in actions
        # Should also pick at least one option index
        assert actions & {0, 1, 2}
