import numpy as np

from yugioh_core.encoding import EVENT_ENTRY_FEATURES, MAX_EVENT_HISTORY
from yugioh_env.game_state import GameState
from yugioh_env.observation import build_observation


def test_event_history_key_present_and_shaped():
    gs = GameState()
    ev = np.zeros((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
    ev[-1, 0] = 70  # msg_type at byte [0]
    obs = build_observation(gs, None, 0, event_history=ev)
    assert obs["event_history"].shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    assert obs["event_history"][-1, 0] == 70


def test_event_history_defaults_to_zeros():
    gs = GameState()
    obs = build_observation(gs, None, 0)
    assert obs["event_history"].shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    assert obs["event_history"].sum() == 0
