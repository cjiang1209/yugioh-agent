"""Tests for configurable evaluation opponents."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from yugioh_rl.config import TrainingConfig
from yugioh_rl.ppo import PPOTrainer


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

class TestParseEvalOpponent:
    def test_greedy(self):
        assert PPOTrainer._parse_eval_opponent("greedy") == ("greedy", "")

    def test_random(self):
        assert PPOTrainer._parse_eval_opponent("random") == ("random", "")

    def test_model_with_path(self):
        assert PPOTrainer._parse_eval_opponent("model:/path/to/ckpt.pt") == (
            "model", "/path/to/ckpt.pt",
        )

    def test_model_relative_path(self):
        assert PPOTrainer._parse_eval_opponent("model:checkpoints/run1/latest.pt") == (
            "model", "checkpoints/run1/latest.pt",
        )


# ---------------------------------------------------------------------------
# Helpers to build a minimal trainer without the full PPOTrainer __init__
# ---------------------------------------------------------------------------

def _dummy_obs():
    return {
        "cards": np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.zeros(20, dtype=np.uint8),
        "actions": np.zeros((32, 12), dtype=np.uint8),
        "action_mask": np.ones(32, dtype=np.int8),
    }


class _FakeNetwork(torch.nn.Module):
    def forward(self, cards, glob, actions, mask):
        B = cards.shape[0]
        return torch.zeros(B, 32), torch.zeros(B)


def _make_trainer_stub(config: TrainingConfig) -> PPOTrainer:
    """Build a PPOTrainer-like object without heavy init (no real network/optimizer)."""
    trainer = object.__new__(PPOTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.network = _FakeNetwork()
    trainer._episode_rewards = []
    trainer._writer = None
    return trainer


# ---------------------------------------------------------------------------
# _evaluate integration (mocked environment)
# ---------------------------------------------------------------------------

class TestEvaluatePassesCheckpoint:
    def test_passes_checkpoint_to_env(self, tmp_path):
        """_evaluate should pass opponent_checkpoint for model specs."""
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            eval_episodes=2,
            eval_opponents=[
                "greedy",
                "model:/fake/checkpoint_100.pt",
                "model:/fake/v2/best.pt",
            ],
        )
        trainer = _make_trainer_stub(config)

        env_calls: list[dict] = []

        class FakeEnv:
            def __init__(self, **kwargs):
                env_calls.append(kwargs)
                self._step = 0

            def reset(self):
                self._step = 0
                return _dummy_obs()

            def step(self, action):
                self._step += 1
                done = self._step >= 2
                info = {"terminal_reward": 1.0} if done else {}
                return _dummy_obs(), 0.0, done, info

            def close(self):
                pass

        with patch("yugioh_rl.ppo.TrainingEnv", FakeEnv):
            trainer._evaluate(config.eval_episodes)

        assert len(env_calls) == 3

        # greedy — no checkpoint key
        assert env_calls[0]["opponent_type"] == "greedy"
        assert "opponent_checkpoint" not in env_calls[0]

        # model entries — checkpoint present
        assert env_calls[1]["opponent_type"] == "model"
        assert env_calls[1]["opponent_checkpoint"] == "/fake/checkpoint_100.pt"

        assert env_calls[2]["opponent_type"] == "model"
        assert env_calls[2]["opponent_checkpoint"] == "/fake/v2/best.pt"

    def test_tensorboard_label_includes_parent_and_stem(self, tmp_path):
        """TensorBoard scalar keys should include parent dir + stem to avoid collisions."""
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            eval_episodes=1,
            eval_opponents=[
                "model:/path/to/checkpoint_100.pt",
                "model:checkpoint_200.pt",  # no parent dir
            ],
        )
        trainer = _make_trainer_stub(config)
        trainer._writer = MagicMock()

        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            def reset(self):
                return _dummy_obs()

            def step(self, action):
                return _dummy_obs(), 0.0, True, {"terminal_reward": 1.0}

            def close(self):
                pass

        with patch("yugioh_rl.ppo.TrainingEnv", FakeEnv):
            trainer._evaluate(1)

        calls = trainer._writer.add_scalar.call_args_list
        keys = [c[0][0] for c in calls]
        # parent "to" + stem "checkpoint_100"
        assert "eval/win_rate_vs_model_to_checkpoint_100" in keys
        # no parent → just stem
        assert "eval/win_rate_vs_model_checkpoint_200" in keys
