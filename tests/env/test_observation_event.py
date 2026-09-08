import numpy as np

from core.encoding import EVENT_ENTRY_FEATURES, EVENT_LAYOUT, MAX_EVENT_HISTORY
from env.game_state import GameState
from env.observation import build_observation


def test_event_history_key_present_and_shaped():
    gs = GameState()
    ev = np.zeros((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
    ev[-1, EVENT_LAYOUT.offsets["msg_type"]] = 70
    obs = build_observation(gs, None, 0, event_history=ev)
    assert obs["event_history"].shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    assert obs["event_history"][-1, 0] == 70


def test_event_history_defaults_to_zeros():
    gs = GameState()
    obs = build_observation(gs, None, 0)
    assert obs["event_history"].shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    assert obs["event_history"].sum() == 0
