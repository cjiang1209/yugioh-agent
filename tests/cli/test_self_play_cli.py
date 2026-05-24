"""Tests for --self-play CLI flags."""

from __future__ import annotations

import sys

from cli.train import _build_fresh_config, parse_args


def test_self_play_flags_flow_into_training_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--total-timesteps",
            "1000",
            "--self-play",
            "--self-play-pool-size",
            "5",
            "--self-play-temperature",
            "0.7",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, save_dir="/tmp/unused")
    assert config.self_play is True
    assert config.self_play_pool_size == 5
    assert config.self_play_temperature == 0.7
