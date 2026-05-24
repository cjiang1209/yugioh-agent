"""Tests for self-play TrainingConfig fields."""

from __future__ import annotations

from yugioh_rl.config import TrainingConfig


def test_self_play_defaults_off() -> None:
    cfg = TrainingConfig()
    assert cfg.self_play is False
    assert cfg.self_play_pool_size == 10
    assert cfg.self_play_temperature == 1.0


def test_self_play_can_be_enabled() -> None:
    cfg = TrainingConfig(self_play=True, self_play_pool_size=5, self_play_temperature=0.7)
    assert cfg.self_play is True
    assert cfg.self_play_pool_size == 5
    assert cfg.self_play_temperature == 0.7
