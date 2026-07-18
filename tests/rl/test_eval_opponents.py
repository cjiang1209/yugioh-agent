"""Tests for the trainer-side eval wrapper.

These pin the contract that ``PPOTrainer._evaluate`` is a faithful, thin
wrapper around ``yugioh_rl.eval.evaluate``: forwarding the right kwargs,
constructing a ``NetworkOpponent`` from ``self.network``, emitting eval
scalars through ``self._sinks``, and toggling ``network.eval()`` /
``.train()`` around the call.

Eval-module internals (TrainingEnv construction, label derivation, win
counting) are covered by ``tests/rl/test_eval_module.py`` so the two layers
fail independently.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.eval import EvalResult
from yugioh_rl.metrics_logging import MultiSink, ScalarMetrics
from yugioh_rl.ppo import PPOTrainer

# ---------------------------------------------------------------------------
# Trainer wrapper integration — patch yugioh_rl.ppo.evaluate_with_agent
# ---------------------------------------------------------------------------


class _RecordingSink:
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)

    def close(self):
        pass


def _make_trainer_stub(config: TrainingConfig) -> PPOTrainer:
    """Build a PPOTrainer-like object without running __init__."""
    trainer = object.__new__(PPOTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.network = MagicMock()
    trainer._episode_rewards = []
    trainer._sinks = MultiSink([_RecordingSink()])
    trainer._deck_pool = [{"main": list(range(1, 41)), "extra": []}]
    trainer._deck_wins = {}
    return trainer


class TestEvaluateWrapper:
    def test_forwards_expected_kwargs_to_evaluate(self, tmp_path):
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            seed=7,
            agent_player="random",
            eval_episodes=5,
            eval_opponents=["greedy", "model:/fake/v1/best.pt"],
        )
        trainer = _make_trainer_stub(config)

        captured: dict = {}

        def _fake_evaluate(agent, **kwargs):
            captured["agent"] = agent
            captured.update(kwargs)
            return []

        with patch("yugioh_rl.ppo.evaluate_with_agent", _fake_evaluate):
            trainer._evaluate(num_episodes=5, global_step=1000)

        assert captured["deck_pool"] is trainer._deck_pool
        assert captured["opponent_specs"] is trainer.config.eval_opponents
        assert captured["num_episodes"] == 5
        assert captured["seed"] == 7 + 999999
        assert captured["agent_player"] == "random"
        # opponent_device deliberately not forwarded so YUGIOH_OPPONENT_DEVICE
        # (and the CPU default) still controls eval-side model opponents.
        assert "opponent_device" not in captured

    def test_wraps_network_in_NetworkOpponent_with_trainer_device(self, tmp_path):
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            eval_episodes=1,
            eval_opponents=["greedy"],
        )
        trainer = _make_trainer_stub(config)

        ctor_calls: list[tuple] = []

        class _FakeNetworkOpponent:
            def __init__(self, network, device: str = "cpu"):
                ctor_calls.append((network, device))

        with (
            patch("yugioh_rl.ppo.NetworkOpponent", _FakeNetworkOpponent),
            patch("yugioh_rl.ppo.evaluate_with_agent", return_value=[]),
        ):
            trainer._evaluate(num_episodes=1, global_step=0)

        assert len(ctor_calls) == 1
        net, device = ctor_calls[0]
        assert net is trainer.network
        assert device == "cpu"

    def test_toggles_network_eval_and_train_around_evaluate(self, tmp_path):
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            eval_episodes=1,
            eval_opponents=["greedy"],
        )
        trainer = _make_trainer_stub(config)

        with patch("yugioh_rl.ppo.evaluate_with_agent", return_value=[]):
            trainer._evaluate(num_episodes=1, global_step=0)

        # network.eval() called before, .train() called after.
        method_names = [c[0] for c in trainer.network.method_calls]
        assert "eval" in method_names
        assert "train" in method_names
        assert method_names.index("eval") < method_names.index("train")

    def test_emits_eval_scalars_to_sink(self, tmp_path):
        config = TrainingConfig(
            save_dir=str(tmp_path / "run"),
            num_envs=1,
            eval_episodes=1,
            eval_opponents=["greedy"],
            deck_paths=["assets/decks/blue_eyes.ydk"],
        )
        trainer = _make_trainer_stub(config)
        results = [EvalResult("greedy", 1, 1, {0: [1.0]})]
        with patch("yugioh_rl.ppo.evaluate_with_agent", return_value=results):
            trainer._evaluate(num_episodes=1, global_step=42)

        recording = trainer._sinks._sinks[0]
        scalar_events = [e for e in recording.events if isinstance(e, ScalarMetrics)]
        assert len(scalar_events) == 1
        ev = scalar_events[0]
        assert ev.global_step == 42
        assert ev.scalars["eval/win_rate/greedy/overall"] == 1.0
        assert ev.scalars["eval/win_rate_by_deck/greedy/blue_eyes"] == 1.0
