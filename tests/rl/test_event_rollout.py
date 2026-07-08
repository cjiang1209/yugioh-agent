import numpy as np

from yugioh_rl.env_wrapper import _obs_to_numpy


class _Obs:
    cards = [[0] * 42] * 200
    global_state = [0] * 21
    actions = [[0] * 28] * 32
    action_mask = [0] * 32
    pending_chain = []
    event_history = [[0] * 30 for _ in range(32)]


def test_obs_to_numpy_includes_event_history():
    d = _obs_to_numpy(_Obs())
    assert d["event_history"].shape == (32, 30)
    assert d["event_history"].dtype == np.uint8
