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


def test_recommend_action_index_passes_obs_through():
    """The recommender receives the observation directly -- there is no
    longer a needs_observation split -- and its choice flows straight
    through as the returned action index."""
    from yugioh_core.encoding import MAX_ACTIONS
    from yugioh_env.models import YuGiOhObservation

    class FakeRecommender:
        def __init__(self):
            self.seen = None

        def select_action(self, obs):
            self.seen = obs
            return 2

    mask = [1, 1, 1] + [0] * (MAX_ACTIONS - 3)
    obs = YuGiOhObservation(action_mask=mask)
    rec = FakeRecommender()
    idx = recommend_action_index(rec, obs)

    assert idx == 2
    assert obs.action_mask[idx] == 1
    assert rec.seen is obs


def test_recommend_action_index_returns_recommenders_choice():
    """recommend_action_index is a thin pass-through: whatever index the
    recommender picks is returned unchanged, for any recommender kind."""
    from yugioh_core.encoding import MAX_ACTIONS
    from yugioh_env.models import YuGiOhObservation

    class FakeGreedy:
        def select_action(self, obs):
            return int(obs.action_mask.sum()) - 1

    obs = YuGiOhObservation(action_mask=[1, 1] + [0] * (MAX_ACTIONS - 2))
    idx = recommend_action_index(FakeGreedy(), obs)

    assert idx == 1


def test_recommender_declines_on_all_zero_mask(monkeypatch) -> None:
    """`_resolve_recommendation` gates on `obs.done` before the mask, and ends
    in a bare `except Exception: return None`. So `done=False` is required to
    reach the mask predicate at all, and the assertion has to be that inference
    was never attempted -- a wrong predicate would blow up on the dummy
    recommender and still return None.
    """
    from types import SimpleNamespace

    from yugioh_env.models import YuGiOhObservation
    from yugioh_env.server import web_api

    called = []
    monkeypatch.setattr(
        web_api,
        "recommend_action_index",
        lambda *a, **k: called.append(1) or 0,
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(recommender=object(), recommend_enabled=True))
    )
    obs = YuGiOhObservation(done=False)  # mask zeros(MAX_ACTIONS), size MAX_ACTIONS
    assert obs.action_mask.size == 32 and not obs.action_mask.any()

    assert web_api._resolve_recommendation(request, obs=obs) is None
    assert not called, "must decline BEFORE attempting inference"
