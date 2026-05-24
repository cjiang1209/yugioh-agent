"""Tests for feature extraction from a checkpoint's TrainingConfig."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yugioh_leaderboard.features import extract_features
from yugioh_rl.config import TrainingConfig


def test_extract_includes_grouping_keys():
    cfg = TrainingConfig(
        rnn_type="lstm",
        rnn_hidden_dim=128,
        reward_shaping=True,
        agent_player="random",
        deck_paths=["assets/decks/blue_eyes.ydk"],
        opponent="greedy",
        total_timesteps=1_000_000,
        seed=42,
        card_embeddings="",
    )
    f = extract_features(cfg)
    assert f["rnn_type"] == "lstm"
    assert f["rnn_hidden_dim"] == 128
    assert f["reward_shaping"] is True
    assert f["agent_player"] == "random"
    assert f["deck_paths"] == ["assets/decks/blue_eyes.ydk"]
    assert f["training_opponent"] == "greedy"
    assert f["total_timesteps"] == 1_000_000
    assert f["seed"] == 42


def test_card_embeddings_symbolic_when_path_empty():
    cfg = TrainingConfig(card_embeddings="")
    assert extract_features(cfg)["card_embeddings"] == "symbolic"


def test_card_embeddings_semantic_when_path_set():
    cfg = TrainingConfig(card_embeddings="assets/embeds.pt")
    assert extract_features(cfg)["card_embeddings"] == "semantic"


def test_features_deterministic_for_same_config():
    cfg = TrainingConfig(rnn_type="gru", seed=7)
    a = extract_features(cfg)
    b = extract_features(cfg)
    assert a == b
    assert list(a.keys()) == list(b.keys())


def test_features_handles_legacy_config_via_normalize(monkeypatch):
    """A pickled config from before a field was added should still extract.

    Reproduces a "legacy checkpoint" by deleting a field from the instance
    __dict__ — this is what unpickled older configs look like.
    """
    cfg = TrainingConfig(rnn_type="lstm")
    del cfg.__dict__["rnn_hidden_dim"]
    f = extract_features(cfg)
    assert f["rnn_hidden_dim"] == 256
