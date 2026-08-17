"""The contract of `Opponent.select_action`: an index, and optionally the
readouts from the forward pass that chose it."""

import pytest

from tests.env.conftest import obs_from_mask
from yugioh_env.opponent import GreedyOpponent, RandomOpponent


@pytest.mark.parametrize(
    "opponent",
    [RandomOpponent(seed=1), GreedyOpponent()],
    ids=["random", "greedy"],
)
def test_opponents_return_an_index_and_an_optional_inference(opponent):
    """Both elements are always present. An opponent with no value head
    reports None rather than a zero-filled Inference, so a consumer can tell
    "nothing to inspect" apart from "evaluated as level"."""
    action, inference = opponent.select_action(obs_from_mask(num_legal=3))

    assert isinstance(action, int)
    assert 0 <= action < 3
    assert inference is None


def test_the_recording_wrapper_forwards_both_elements():
    """RecordingOpponent must not swallow the inference: recording a duel
    cannot change what the caller sees."""
    from yugioh_env.opponent import Inference
    from yugioh_env.replay import GameRecording, RecordingOpponent

    class FakeInner:
        needs_board_state = False

        def select_action(self, obs):
            return 1, Inference(value=0.5, action_probs=[0.4, 0.6])

        def reseed(self, seed):
            pass

    wrapper = RecordingOpponent(FakeInner(), GameRecording(setup={}), seat_fn=lambda: 1)
    action, inference = wrapper.select_action(obs_from_mask(num_legal=2))

    assert action == 1
    assert inference is not None and inference.value == 0.5
