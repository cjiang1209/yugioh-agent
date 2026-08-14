"""Unit tests for the web recommender loader + obs->index helper."""

import pytest

from yugioh_env.server.recommender import (
    make_recommender,
    recommend,
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


def test_recommend_passes_obs_through():
    """The recommender receives the observation directly, and its choice
    flows straight through as the returned action index."""
    from yugioh_env.models import Pass, YuGiOhObservation

    class FakeRecommender:
        def __init__(self):
            self.seen = None

        def select_action(self, obs):
            self.seen = obs
            return 2, None

    obs = YuGiOhObservation(action_descriptors=[Pass() for _ in range(3)])
    rec = FakeRecommender()
    result = recommend(rec, obs)

    assert result.action_index == 2
    assert result.action_index < obs.num_actions
    assert rec.seen is obs


def test_recommend_returns_recommenders_choice():
    """recommend is a thin pass-through: whatever index the recommender picks
    is returned unchanged, for any recommender kind."""
    from yugioh_env.models import Pass, YuGiOhObservation

    class FakeGreedy:
        def select_action(self, obs):
            return obs.num_actions - 1, None

    obs = YuGiOhObservation(action_descriptors=[Pass(), Pass()])
    result = recommend(FakeGreedy(), obs)

    assert result.action_index == 1


def test_recommend_carries_the_value_head_readout_when_present():
    from yugioh_env.models import Pass, YuGiOhObservation
    from yugioh_env.opponent import Inference

    class FakeNetworkRecommender:
        def select_action(self, obs):
            return 0, Inference(value=-0.25, action_probs=[0.7, 0.3])

    obs = YuGiOhObservation(action_descriptors=[Pass(), Pass()])
    result = recommend(FakeNetworkRecommender(), obs)

    assert result.action_index == 0
    assert result.value == -0.25
    assert result.action_probs == [0.7, 0.3]


def test_recommend_reports_no_readout_without_a_value_head():
    """random / greedy / ygo-agent pick an index but have nothing to inspect,
    so both readouts must be None rather than zero."""
    from yugioh_env.models import Pass, YuGiOhObservation

    class FakeGreedy:
        def select_action(self, obs):
            return 1, None

    obs = YuGiOhObservation(action_descriptors=[Pass(), Pass()])
    result = recommend(FakeGreedy(), obs)

    assert result.action_index == 1
    assert result.value is None
    assert result.action_probs is None


def test_recommender_declines_when_there_are_no_actions(monkeypatch) -> None:
    """`_resolve_recommendation` gates on `obs.done` before the action count,
    and ends in a bare `except Exception: return None`. So `done=False` is
    required to reach that predicate at all, and the assertion has to be that
    inference was never attempted -- a wrong predicate would blow up on the
    dummy recommender and still return None.
    """
    from types import SimpleNamespace

    from yugioh_env.models import YuGiOhObservation
    from yugioh_env.server import web_api

    called = []
    monkeypatch.setattr(
        web_api,
        "recommend",
        lambda *a, **k: called.append(1) or 0,
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(recommender=object(), recommend_enabled=True))
    )
    obs = YuGiOhObservation(done=False)  # no prompt, so no descriptors
    assert obs.num_actions == 0

    assert web_api._resolve_recommendation(request, obs=obs) is None
    assert not called, "must decline BEFORE attempting inference"
