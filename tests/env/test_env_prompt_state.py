"""Regression tests for YuGiOhEnvironment.current_msg / num_actions invariants.

These properties are consumed by the eval driver (Phase 2+), which needs them
to stay consistent with the observation's action_mask across:

1. **Multi-step card selection** — step() accumulates picks without
   re-entering _process_to_agent_choice; _current_msg must track the updated
   prompt so external readers see the narrowed state (not the original msg).
2. **Terminal transitions** — after done=True, _current_msg must be cleared
   so num_actions returns 0 (matching the terminal observation's all-zero
   action_mask).

Constructed via object.__new__ so these are pure unit tests — no engine deps.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    CHAIN_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_PENDING_CHAIN,
)
from yugioh_env.action_space import ActionMapper
from yugioh_env.models import YuGiOhAction
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment


@pytest.fixture(autouse=True)
def _stub_build_observation(monkeypatch):
    """Replace build_observation with a fast stub — we don't care about obs
    contents here, only about prompt-state bookkeeping around step() and
    _make_terminal_observation."""
    import yugioh_env.server.yugioh_environment as env_mod

    monkeypatch.setattr(
        env_mod,
        "build_observation",
        lambda gs, msg, agent_player, query_fn=None: {
            "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            "global_state": np.zeros(GLOBAL_FEATURES, dtype=np.uint8),
            "pending_chain": np.zeros(
                (MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES), dtype=np.uint8
            ),
        },
    )


def _bare_env() -> YuGiOhEnvironment:
    """Build an env instance without running __init__ (no libocgcore / cards.cdb needed)."""
    env = object.__new__(YuGiOhEnvironment)
    env._duel = MagicMock()
    env._duel.is_finished = False
    env._mapper = ActionMapper()
    env._current_msg = None
    env._card_sel = []
    env._step_count = 0
    env._last_frames = []
    env._agent_player = 0
    env._card_db = None
    env._collapse_forced = False
    return env


# ---------------------------------------------------------------------------
# num_actions returns 0 when no prompt is active
# ---------------------------------------------------------------------------


def test_num_actions_zero_when_no_active_prompt():
    env = _bare_env()
    assert env.current_msg is None
    assert env.num_actions == 0


def test_num_actions_matches_mapper_when_prompt_active():
    env = _bare_env()
    # Simulate a fresh prompt the way _process_to_agent_choice would: set msg
    # and update mapper. Use MSG_SELECT_YESNO for a simple 2-action prompt.
    from yugioh_core.constants import MSG_SELECT_YESNO

    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    env._current_msg = msg
    env._mapper.update(msg)
    assert env.num_actions == env._mapper.num_actions
    assert env.num_actions > 0


# ---------------------------------------------------------------------------
# Multi-step card selection keeps current_msg in sync with the mapper
# ---------------------------------------------------------------------------


def test_multi_step_updates_current_msg_in_sync_with_mapper():
    """In the response-is-None branch, _current_msg should track _selected picks."""
    env = _bare_env()

    # Pre-populate state as if _process_to_agent_choice just ran.
    original_msg = {"msg_type": 15, "player": 0, "cards": [1, 2, 3], "min": 2, "max": 2}
    env._current_msg = dict(original_msg)

    # Stub _mapper to report non-zero actions and yield response=None for the
    # first pick (simulating an intermediate step in multi-card selection).
    env._mapper = MagicMock()
    env._mapper.action_to_response = MagicMock(return_value=None)
    env._mapper.get_action_index = MagicMock(return_value=0)
    env._mapper.num_actions = 3
    env._mapper.get_action_mask = MagicMock(
        return_value=np.array([1, 1, 1] + [0] * 29, dtype=np.int8)
    )
    env._mapper.get_action_features = MagicMock(
        return_value=np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
    )

    # Execute step in the multi-step branch (response is None).
    env.step(YuGiOhAction(action_index=0))

    # After the step, _current_msg should reflect the accumulated selection.
    assert env._current_msg is not None
    assert "_selected" in env._current_msg
    assert env._current_msg["_selected"] == [0]
    # And _mapper.update should have been called with that same updated msg.
    env._mapper.update.assert_called_once()
    called_with = env._mapper.update.call_args[0][0]
    assert called_with["_selected"] == [0]
    # num_actions reflects the (stubbed) mapper state, consistent with the new prompt.
    assert env.num_actions == 3


# ---------------------------------------------------------------------------
# Terminal observation clears the prompt state
# ---------------------------------------------------------------------------


def test_terminal_observation_clears_current_msg():
    env = _bare_env()
    env._current_msg = {"msg_type": 11, "player": 0}  # stale prompt
    env._card_sel = [1, 2]
    env._mapper.update({"msg_type": 11, "player": 0})  # mapper has state

    # Provide duel.game_state for reward computation.
    env._duel.game_state.is_finished = True
    env._duel.game_state.winner = 0

    obs = env._make_terminal_observation()

    assert obs.done is True
    assert env.current_msg is None
    assert env.num_actions == 0
    assert env._card_sel == []
    # And the terminal observation's action_mask is empty (consistent with actions=[]).
    assert obs.action_mask == []
