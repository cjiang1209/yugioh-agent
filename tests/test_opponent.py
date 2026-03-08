"""Tests for opponent policies and seed determinism."""

import random

from yugioh_env.action_space import ActionMapper
from yugioh_env.constants import MSG_SELECT_YESNO
from yugioh_env.opponent import GreedyOpponent, RandomOpponent


def _make_yesno_mapper() -> ActionMapper:
    """Create an ActionMapper with a simple yes/no message (2 actions)."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    return mapper


def test_random_opponent_deterministic_with_seed():
    """Same seed should produce identical action sequences."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    results = []
    for _ in range(2):
        opp = RandomOpponent(seed=42)
        mapper = _make_yesno_mapper()
        actions = [opp.select_action(msg, mapper) for _ in range(20)]
        results.append(actions)
    assert results[0] == results[1]


def test_random_opponent_reseed_restores_determinism():
    """Calling reseed() should reset the RNG to produce the same sequence."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    opp = RandomOpponent(seed=99)
    mapper = _make_yesno_mapper()

    # Generate a sequence
    run1 = [opp.select_action(msg, mapper) for _ in range(20)]

    # Reseed and generate again
    opp.reseed(99)
    run2 = [opp.select_action(msg, mapper) for _ in range(20)]

    assert run1 == run2


def test_random_opponent_different_seeds_differ():
    """Different seeds should (almost certainly) produce different sequences."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    mapper = _make_yesno_mapper()

    opp1 = RandomOpponent(seed=1)
    opp2 = RandomOpponent(seed=2)
    run1 = [opp1.select_action(msg, mapper) for _ in range(50)]
    run2 = [opp2.select_action(msg, mapper) for _ in range(50)]

    assert run1 != run2


def test_greedy_opponent_reseed_is_noop():
    """GreedyOpponent.reseed() should not raise."""
    opp = GreedyOpponent()
    opp.reseed(42)  # should be a no-op


def test_pick_action_random_seeded():
    """Client-side pick_action_random is deterministic when random module is seeded."""
    # Import here to avoid polluting module-level random state
    from cli.play_client import pick_action_random
    from yugioh_env.models import YuGiOhObservation

    mask = [1, 1, 1, 1, 0, 0, 0, 0] + [0] * 24  # 4 legal actions
    obs = YuGiOhObservation(
        cards=[],
        global_state=[0] * 20,
        actions=[[0] * 12] * 32,
        action_mask=mask,
        done=False,
        reward=0.0,
    )

    results = []
    for _ in range(2):
        random.seed(123)
        actions = [pick_action_random(obs) for _ in range(20)]
        results.append(actions)

    assert results[0] == results[1]
