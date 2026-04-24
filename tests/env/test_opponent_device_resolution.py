"""Tests for yugioh_env.server.yugioh_environment._resolve_opponent_device."""

from __future__ import annotations

import os

import pytest

from yugioh_env.server.yugioh_environment import _resolve_opponent_device


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Every test starts with YUGIOH_OPPONENT_DEVICE unset."""
    monkeypatch.delenv("YUGIOH_OPPONENT_DEVICE", raising=False)


def test_default_is_cpu():
    """No config key, no env var → 'cpu'."""
    assert _resolve_opponent_device({}) == "cpu"


def test_env_var_used_when_config_missing(monkeypatch):
    monkeypatch.setenv("YUGIOH_OPPONENT_DEVICE", "cuda")
    assert _resolve_opponent_device({}) == "cuda"


def test_config_key_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("YUGIOH_OPPONENT_DEVICE", "cuda")
    assert _resolve_opponent_device({"opponent_device": "cpu"}) == "cpu"


def test_config_key_used_when_env_var_missing():
    assert _resolve_opponent_device({"opponent_device": "cuda"}) == "cuda"


def test_empty_config_key_falls_back_to_env_var(monkeypatch):
    """An empty string in config should not mask the env var (or: caller passed empty, treat as unset)."""
    monkeypatch.setenv("YUGIOH_OPPONENT_DEVICE", "cuda")
    # The ``or`` operator treats "" as falsy, so env var wins — this matches the
    # pre-refactor behavior at yugioh_environment.py:135.
    assert _resolve_opponent_device({"opponent_device": ""}) == "cuda"


def test_none_config_key_falls_back_to_default():
    """A None config value should fall through to the env var / cpu default."""
    assert _resolve_opponent_device({"opponent_device": None}) == "cpu"
