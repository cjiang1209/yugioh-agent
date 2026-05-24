"""Tests for yugioh_env.opponent.make_opponent / parse_opponent_spec."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yugioh_env.opponent import (
    GreedyOpponent,
    RandomOpponent,
    make_opponent,
    parse_opponent_spec,
)

# ---------------------------------------------------------------------------
# parse_opponent_spec
# ---------------------------------------------------------------------------


class TestParseOpponentSpec:
    def test_greedy(self):
        assert parse_opponent_spec("greedy") == ("greedy", "")

    def test_random(self):
        assert parse_opponent_spec("random") == ("random", "")

    def test_model_with_path(self):
        assert parse_opponent_spec("model:/p/ckpt.pt") == ("model", "/p/ckpt.pt")

    def test_model_relative_path(self):
        assert parse_opponent_spec("model:checkpoints/run1/latest.pt") == (
            "model",
            "checkpoints/run1/latest.pt",
        )

    def test_model_empty_path(self):
        """parse does not validate; empty path is preserved for caller to reject."""
        assert parse_opponent_spec("model:") == ("model", "")


# ---------------------------------------------------------------------------
# make_opponent
# ---------------------------------------------------------------------------


class TestMakeOpponent:
    def test_greedy(self):
        opp = make_opponent("greedy")
        assert isinstance(opp, GreedyOpponent)

    def test_random(self):
        opp = make_opponent("random", seed=0)
        assert isinstance(opp, RandomOpponent)

    def test_random_deterministic_with_seed(self):
        """Same seed produces identical RandomOpponent behavior."""
        msg = {"msg_type": 0}
        opp_a = make_opponent("random", seed=123)
        opp_b = make_opponent("random", seed=123)
        # num_actions=4 so select_action returns randint(0, 3)
        seq_a = [opp_a.select_action(msg, 4) for _ in range(20)]
        seq_b = [opp_b.select_action(msg, 4) for _ in range(20)]
        assert seq_a == seq_b

    def test_model_empty_path_raises(self):
        with pytest.raises(ValueError, match="checkpoint path"):
            make_opponent("model:")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown opponent"):
            make_opponent("bogus")

    def test_model_forwards_device_and_path(self):
        """make_opponent('model:path', device='cuda') forwards to ModelOpponent ctor."""
        captured: dict = {}

        class _FakeModelOpponent:
            def __init__(self, checkpoint_path: str, device: str = "cpu"):
                captured["checkpoint_path"] = checkpoint_path
                captured["device"] = device

        with patch("yugioh_env.opponent.ModelOpponent", _FakeModelOpponent):
            make_opponent("model:/some/ckpt.pt", device="cuda")

        assert captured == {"checkpoint_path": "/some/ckpt.pt", "device": "cuda"}

    def test_model_missing_file_raises(self, tmp_path):
        """make_opponent with a nonexistent model path propagates the torch error.

        We don't pin the exact exception type (torch changes it between versions);
        we only verify that make_opponent does not silently succeed.
        """
        pytest.importorskip("torch")
        missing = tmp_path / "nope.pt"
        with pytest.raises(Exception):  # noqa: B017 — error class varies by torch version
            make_opponent(f"model:{missing}")
