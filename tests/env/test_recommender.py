"""Unit tests for the web recommender loader + obs->index helper."""

import pytest

from yugioh_env.server.recommender import (
    make_recommender,
    recommend_action_index,
    recommender_device_from_env,
    recommender_spec_from_env,
)


def test_make_recommender_none_when_spec_falsy():
    assert make_recommender(None) is None
    assert make_recommender("") is None


def test_make_recommender_builds_non_model_opponents():
    from yugioh_env.opponent import GreedyOpponent, RandomOpponent

    assert isinstance(make_recommender("random"), RandomOpponent)
    assert isinstance(make_recommender("greedy"), GreedyOpponent)


def test_make_recommender_rejects_unknown_spec():
    with pytest.raises(ValueError):
        make_recommender("not-a-real-opponent")


def test_make_recommender_rejects_empty_model_path():
    with pytest.raises(ValueError, match="model"):
        make_recommender("model:")


def test_spec_from_env_reads_variable(monkeypatch):
    monkeypatch.delenv("YUGIOH_RECOMMENDER", raising=False)
    assert recommender_spec_from_env() is None
    monkeypatch.setenv("YUGIOH_RECOMMENDER", "greedy")
    assert recommender_spec_from_env() == "greedy"
    monkeypatch.setenv("YUGIOH_RECOMMENDER", "")
    assert recommender_spec_from_env() is None


def test_device_from_env_default(monkeypatch):
    monkeypatch.delenv("YUGIOH_RECOMMENDER_DEVICE", raising=False)
    assert recommender_device_from_env() == "cpu"
    monkeypatch.setenv("YUGIOH_RECOMMENDER_DEVICE", "cuda")
    assert recommender_device_from_env() == "cuda"


class _FakeEnv:
    def __init__(self, msg, num_actions):
        self.current_msg = msg
        self.num_actions = num_actions


def test_recommend_action_index_network_gets_obs():
    """A needs_observation recommender receives numpy obs arrays; its index is returned."""
    import numpy as np

    from yugioh_core.encoding import CHAIN_ENTRY_FEATURES, MAX_PENDING_CHAIN
    from yugioh_env.models import YuGiOhObservation

    class FakeNet:
        needs_observation = True

        def __init__(self):
            self.seen = None

        def set_observation(self, obs_dict):
            self.seen = obs_dict

        def select_action(self, msg, num_actions):
            return 2

    obs = YuGiOhObservation(
        cards=[[0, 0]],
        global_state=[0, 0],
        actions=[[0], [0], [0]],
        action_mask=[1, 1, 1],
        pending_chain=[[0] * CHAIN_ENTRY_FEATURES for _ in range(MAX_PENDING_CHAIN)],
        event_history=[[0]],
    )
    rec = FakeNet()
    idx = recommend_action_index(rec, _FakeEnv({"msg_type": 11}, 3), obs)

    assert idx == 2
    assert obs.action_mask[idx] == 1
    assert isinstance(rec.seen["cards"], np.ndarray)
    assert rec.seen["action_mask"].tolist() == [1, 1, 1]
    assert isinstance(rec.seen["pending_chain"], np.ndarray)
    assert rec.seen["pending_chain"].shape == (MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)


def test_recommend_action_index_non_network_skips_obs():
    """A recommender that doesn't need obs is never handed one, and gets the real msg."""
    from yugioh_env.models import YuGiOhObservation

    class FakeGreedy:
        needs_observation = False

        def __init__(self):
            self.set_obs_called = False
            self.msg_seen = None
            self.num_seen = None

        def set_observation(self, obs_dict):
            self.set_obs_called = True

        def select_action(self, msg, num_actions):
            self.msg_seen = msg
            self.num_seen = num_actions
            return num_actions - 1

    obs = YuGiOhObservation(
        cards=[[0]],
        global_state=[0],
        actions=[[0], [0]],
        action_mask=[1, 1],
        event_history=[[0]],
    )
    rec = FakeGreedy()
    idx = recommend_action_index(rec, _FakeEnv({"msg_type": 22}, 2), obs)

    assert rec.set_obs_called is False
    assert rec.msg_seen == {"msg_type": 22}
    assert rec.num_seen == 2
    assert idx == 1
