"""Mechanism tests for the `needs_board_state` capability hint.

Whether each opponent's declaration is truthful is not asserted here -- the
opponent set is small and reviewable. What is covered:

1. `_build_seat_observation(..., include_board=False)` shapes the board
   fields as zeros while fully populating the prompt side.
2. The core gates `include_board` on the *installed opponent's*
   `needs_board_state`, not a hardcoded value.
3. `ModelOpponent` and `RecordingOpponent` delegate the flag to the
   wrapped/inner opponent.
"""

from __future__ import annotations

import random

import pytest

from yugioh_core.encoding import (
    CARD_FEATURES,
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
)
from yugioh_env.models import YuGiOhAction
from yugioh_env.opponent import Inference, ModelOpponent, Opponent, RandomOpponent
from yugioh_env.replay import GameRecording, RecordingOpponent

# ---------------------------------------------------------------------------
# 1. include_board=False shapes the board fields as zeros
# ---------------------------------------------------------------------------


def test_include_board_false_zeros_board_but_fully_populates_prompt(
    lib, db_path, script_dirs
) -> None:
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    try:
        obs_full = env.reset(seed=42)
        # Sanity: the real board is non-empty right after reset (decks/hands
        # dealt), so a False cards.any() below is a genuine signal, not a
        # fluke of an empty board.
        assert obs_full.cards.any()
        assert obs_full.global_state.any()

        obs_no_board = env._build_seat_observation(env._mapper, include_board=False)

        # Board fields fall back to shaped zeros.
        assert obs_no_board.cards.shape == (MAX_CARDS, CARD_FEATURES)
        assert not obs_no_board.cards.any()
        assert obs_no_board.global_state.shape == (GLOBAL_FEATURES,)
        assert not obs_no_board.global_state.any()
        assert obs_no_board.pending_chain.shape == (MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)
        assert obs_no_board.event_history.shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)

        # Prompt side is fully populated and identical to the full build.
        assert obs_no_board.action_mask.tolist() == obs_full.action_mask.tolist()
        assert obs_no_board.action_descriptors == obs_full.action_descriptors
        assert obs_no_board.prompt_meta == obs_full.prompt_meta
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 2. The core passes include_board = opponent.needs_board_state
# ---------------------------------------------------------------------------


class _SpyOpponent(RandomOpponent):
    """Plays like RandomOpponent, but its needs_board_state is settable."""

    def __init__(self, needs_board_state: bool, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._needs_board_state = needs_board_state

    @property
    def needs_board_state(self) -> bool:
        return self._needs_board_state


def _install_observation_spy(env, calls: list[bool]) -> None:
    """Wrap env._build_seat_observation to record explicit include_board kwargs.

    Recording only the explicit ones isolates the opponent call site: the
    agent seat (`_make_observation`) relies on the default, so it cannot
    pollute the recording.
    """
    original = env._build_seat_observation

    def spy(*args, **kwargs):
        if "include_board" in kwargs:
            calls.append(kwargs["include_board"])
        return original(*args, **kwargs)

    env._build_seat_observation = spy


def _play_until_done_or(env, max_steps: int = 60) -> None:
    obs = env.reset(seed=7)
    rng = random.Random(1)
    steps = 0
    while not obs.done and steps < max_steps:
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))
        steps += 1


@pytest.mark.parametrize("needs_board", [False, True])
def test_core_passes_installed_opponents_hint_as_include_board(
    lib, db_path, script_dirs, needs_board
) -> None:
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    try:
        calls: list[bool] = []
        env.set_opponent(_SpyOpponent(needs_board_state=needs_board))
        _install_observation_spy(env, calls)
        _play_until_done_or(env)

        assert calls, "opponent seat was never asked for a multi-action decision"
        assert all(c is needs_board for c in calls)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 3. ModelOpponent / RecordingOpponent delegate needs_board_state
# ---------------------------------------------------------------------------


class _FakeInner(Opponent):
    def __init__(self, needs_board_state: bool) -> None:
        self._needs_board_state = needs_board_state

    @property
    def needs_board_state(self) -> bool:
        return self._needs_board_state

    def select_action(self, obs) -> tuple[int, Inference | None]:
        return 0, None


@pytest.mark.parametrize("inner_hint", [True, False])
def test_model_opponent_delegates_needs_board_state(inner_hint) -> None:
    # Bypass __init__ (loads a real checkpoint via torch) -- delegation is
    # a pure property forward, independent of construction.
    mo = ModelOpponent.__new__(ModelOpponent)
    mo._impl = _FakeInner(inner_hint)
    assert mo.needs_board_state is inner_hint


@pytest.mark.parametrize("inner_hint", [True, False])
def test_recording_opponent_delegates_needs_board_state(inner_hint) -> None:
    recording = GameRecording(setup={"agent_player": 0})
    wrapped = RecordingOpponent(_FakeInner(inner_hint), recording, seat_fn=lambda: 1)
    assert wrapped.needs_board_state is inner_hint
