"""Tests for yugioh_rl.eval — core behavior of the standalone eval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_env.opponent import (
    GreedyOpponent,
    Opponent,
    RandomOpponent,
)
from yugioh_rl.eval import (
    EvalResult,
    evaluate_with_agent,
    log_results_to_tensorboard,
    make_eval_agent,
    opponent_label_from_spec,
    run_match,
)

# ---------------------------------------------------------------------------
# opponent_label_from_spec — pinned to the pre-refactor TensorBoard label format
# ---------------------------------------------------------------------------


class TestOpponentLabelFromSpec:
    def test_greedy(self):
        assert opponent_label_from_spec("greedy") == "greedy"

    def test_random(self):
        assert opponent_label_from_spec("random") == "random"

    def test_model_no_parent(self):
        # "model:foo.pt" — Path("foo.pt").parent.name is "" so just stem.
        assert opponent_label_from_spec("model:foo.pt") == "model_foo"

    def test_model_with_parent(self):
        assert opponent_label_from_spec("model:/a/b/c/latest.pt") == "model_c_latest"

    def test_model_relative_with_parent(self):
        assert opponent_label_from_spec("model:checkpoints/run1/latest.pt") == ("model_run1_latest")


# ---------------------------------------------------------------------------
# make_eval_agent
# ---------------------------------------------------------------------------


class TestMakeEvalAgent:
    def test_random(self):
        agent = make_eval_agent("random", seed=0)
        assert isinstance(agent, RandomOpponent)

    def test_greedy(self):
        agent = make_eval_agent("greedy")
        assert isinstance(agent, GreedyOpponent)

    def test_model_forwards_to_model_opponent(self):
        captured: dict = {}

        class _FakeModelOpponent:
            def __init__(self, checkpoint_path: str, device: str = "cpu"):
                captured["path"] = checkpoint_path
                captured["device"] = device

        # make_eval_agent delegates to make_opponent, which constructs
        # ModelOpponent from yugioh_env.opponent — that's the patch target.
        with patch("yugioh_env.opponent.ModelOpponent", _FakeModelOpponent):
            make_eval_agent("model:/p/ckpt.pt", device="cuda")

        assert captured == {"path": "/p/ckpt.pt", "device": "cuda"}

    def test_model_empty_path_raises(self):
        with pytest.raises(ValueError, match="checkpoint path"):
            make_eval_agent("model:")

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError, match="unknown opponent"):
            make_eval_agent("bogus")

    def test_network_overrides_spec(self):
        """When a network is provided, spec is ignored and a NetworkOpponent is built."""
        captured: dict = {}

        class _FakeNetworkOpponent:
            def __init__(self, network, device: str = "cpu"):
                captured["network"] = network
                captured["device"] = device

        sentinel_net = object()
        with patch("yugioh_rl.eval.NetworkOpponent", _FakeNetworkOpponent):
            make_eval_agent("greedy", network=sentinel_net, device="cuda")

        assert captured == {"network": sentinel_net, "device": "cuda"}


# ---------------------------------------------------------------------------
# run_match — fake env + fake agent, no monkeypatching
# ---------------------------------------------------------------------------


class _RecordingAgent(Opponent):
    """Records every call so tests can assert the driver contract."""

    def __init__(self, *, needs_obs: bool = False):
        self._needs_obs = needs_obs
        self.reseed_calls: list[int] = []
        self.set_observation_calls: list[dict] = []
        self.select_action_calls: list[tuple[dict | None, int]] = []

    @property
    def needs_observation(self) -> bool:
        return self._needs_obs

    def reseed(self, seed: int) -> None:
        self.reseed_calls.append(seed)

    def set_observation(self, obs):
        self.set_observation_calls.append(obs)

    def select_action(self, msg, num_actions):
        self.select_action_calls.append((msg, num_actions))
        return 0


class _ScriptedEnv:
    """Fake TrainingEnv driven by a list of scripted step() outcomes per episode.

    Each episode is a list of dicts: ``{"done": bool, "reward": float,
    "agent_deck_idx": int}``. Matches the post-refactor ``TrainingEnv``
    contract:
    - ``reset()`` advances to the next scripted episode and is called
      **explicitly** by the caller before each episode.
    - ``step()`` returns the terminal obs on ``done=True``; it does NOT
      auto-advance the episode pointer.
    """

    def __init__(self, scripts: list[list[dict]]):
        self._scripts = scripts
        self._ep = -1
        self._step = 0
        self.reset_calls = 0
        self.current_msg = {"msg_type": 0}
        self.num_actions = 4

    def reset(self, *, episode_idx: int | None = None):
        self.reset_calls += 1
        if episode_idx is not None:
            self._ep = episode_idx - 1  # 1-indexed: matches TrainingEnv
        else:
            self._ep += 1
        self._step = 0
        return _dummy_obs()

    def step(self, action):
        outcome = self._scripts[self._ep][self._step]
        self._step += 1
        info = {}
        done = bool(outcome.get("done"))
        if done:
            info["terminal_reward"] = outcome.get("reward", 0.0)
            info["agent_deck_idx"] = outcome.get("agent_deck_idx", 0)
            # No auto-advance — caller is responsible for the next reset.
        return _dummy_obs(), 0.0, done, info


def _dummy_obs() -> dict[str, np.ndarray]:
    return {
        "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
        "global_state": np.zeros(GLOBAL_FEATURES, dtype=np.uint8),
        "actions": np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8),
        "action_mask": np.ones(32, dtype=np.int8),
    }


class TestRunMatch:
    def test_counts_wins_and_per_deck(self):
        agent = _RecordingAgent()
        # 4 episodes: win/loss/win/win, decks 0/0/1/1.
        env = _ScriptedEnv(
            [
                [{"done": True, "reward": 1.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": -1.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": 1.0, "agent_deck_idx": 1}],
                [{"done": True, "reward": 1.0, "agent_deck_idx": 1}],
            ]
        )
        wins, per_deck = run_match(agent, env, num_episodes=4, base_seed=42)
        assert wins == 3
        assert per_deck == {0: [1.0, 0.0], 1: [1.0, 1.0]}

    def test_reseeds_agent_per_episode(self):
        """run_match(base_seed=S) calls agent.reseed(S+i+1) before episode i."""
        agent = _RecordingAgent()
        env = _ScriptedEnv(
            [
                [{"done": True, "reward": 1.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": 0.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": 1.0, "agent_deck_idx": 0}],
            ]
        )
        run_match(agent, env, num_episodes=3, base_seed=100)
        assert agent.reseed_calls == [101, 102, 103]

    def test_passes_env_state_to_agent(self):
        """Each select_action call receives env.current_msg + env.num_actions."""
        agent = _RecordingAgent()
        env = _ScriptedEnv([[{"done": True, "reward": 1.0, "agent_deck_idx": 0}]])
        env.current_msg = {"msg_type": 11, "test": "marker"}
        env.num_actions = 7
        run_match(agent, env, num_episodes=1, base_seed=0)
        assert agent.select_action_calls == [({"msg_type": 11, "test": "marker"}, 7)]

    def test_set_observation_called_when_needs_obs(self):
        agent = _RecordingAgent(needs_obs=True)
        env = _ScriptedEnv([[{"done": True, "reward": 1.0, "agent_deck_idx": 0}]])
        run_match(agent, env, num_episodes=1, base_seed=0)
        assert len(agent.set_observation_calls) == 1
        # And the obs is the dict from env.reset() / env.step()
        assert "cards" in agent.set_observation_calls[0]

    def test_set_observation_skipped_when_not_needs_obs(self):
        agent = _RecordingAgent(needs_obs=False)
        env = _ScriptedEnv([[{"done": True, "reward": 1.0, "agent_deck_idx": 0}]])
        run_match(agent, env, num_episodes=1, base_seed=0)
        assert agent.set_observation_calls == []

    def test_zero_episodes_skips_reset(self):
        """num_episodes == 0 must not call env.reset() — disabling eval should
        not pay a duel-init cost (or trigger reset-time failures)."""
        agent = _RecordingAgent()
        env = _ScriptedEnv([])
        wins, per_deck = run_match(agent, env, num_episodes=0, base_seed=42)
        assert wins == 0
        assert per_deck == {}
        assert env.reset_calls == 0
        assert agent.reseed_calls == []

    def test_resets_explicitly_per_episode(self):
        """run_match calls env.reset() once per episode now that
        TrainingEnv.step() no longer auto-resets.  Without this, the
        terminal obs of episode N would be fed to the agent at the start
        of episode N+1 — bug.
        """
        agent = _RecordingAgent()
        env = _ScriptedEnv(
            [
                [{"done": True, "reward": 1.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": 0.0, "agent_deck_idx": 0}],
                [{"done": True, "reward": 1.0, "agent_deck_idx": 0}],
            ]
        )
        run_match(agent, env, num_episodes=3, base_seed=0)
        assert env.reset_calls == 3
        assert agent.reseed_calls == [1, 2, 3]


# ---------------------------------------------------------------------------
# evaluate — orchestration over multiple opponents
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_training_env_factory():
    """Provide a per-test ``_FakeTrainingEnv`` class whose instances are tracked
    in a fresh list. Avoids cross-test bleed from a class-level mutable list.
    """
    instances: list = []

    class _FakeTrainingEnv:
        """Records ctor kwargs and yields a 1-episode-win-at-deck-0 outcome."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self._step_count = 0
            self.current_msg = {"msg_type": 0}
            self.num_actions = 4
            instances.append(self)

        def reset(self, *, episode_idx: int | None = None):
            self._step_count = 0
            return _dummy_obs()

        def step(self, action):
            self._step_count += 1
            return (
                _dummy_obs(),
                0.0,
                True,
                {"terminal_reward": 1.0, "agent_deck_idx": 0},
            )

        def close(self):
            self.closed = True

    return _FakeTrainingEnv, instances


class TestEvaluate:
    def test_constructs_one_env_per_spec_with_correct_kwargs(self, fake_training_env_factory):
        FakeEnv, instances = fake_training_env_factory
        agent = _RecordingAgent()
        with patch("yugioh_rl.eval.TrainingEnv", FakeEnv):
            results = evaluate_with_agent(
                agent,
                deck_pool=[{"main": list(range(40)), "extra": []}],
                opponent_specs=["greedy", "random", "model:/p/v1.pt"],
                num_episodes=2,
                seed=42,
                agent_player="random",
            )

        assert len(results) == 3
        assert len(instances) == 3
        assert all(env.closed for env in instances)

        for env, expected_spec in zip(
            instances, ["greedy", "random", "model:/p/v1.pt"], strict=True
        ):
            assert env.kwargs["opponent"] == expected_spec
            assert env.kwargs["reward_shaping"] is False
            assert env.kwargs["seed"] == 42
            assert env.kwargs["agent_player"] == "random"
            # opponent_device omitted → kwarg absent so env-var fallback wins.
            assert "opponent_device" not in env.kwargs

    def test_opponent_device_forwarded_only_when_provided(self, fake_training_env_factory):
        FakeEnv, instances = fake_training_env_factory
        agent = _RecordingAgent()
        with patch("yugioh_rl.eval.TrainingEnv", FakeEnv):
            evaluate_with_agent(
                agent,
                deck_pool=[{"main": list(range(40)), "extra": []}],
                opponent_specs=["greedy"],
                num_episodes=1,
                seed=0,
                opponent_device="cuda",
            )
        assert instances[0].kwargs["opponent_device"] == "cuda"

    def test_results_carry_label_and_per_deck(self, fake_training_env_factory):
        FakeEnv, _ = fake_training_env_factory
        agent = _RecordingAgent()
        with patch("yugioh_rl.eval.TrainingEnv", FakeEnv):
            results = evaluate_with_agent(
                agent,
                deck_pool=[{"main": list(range(40)), "extra": []}],
                opponent_specs=["greedy", "model:/p/v1.pt"],
                num_episodes=2,
                seed=0,
            )

        assert results[0].opponent_label == "greedy"
        assert results[0].episodes == 2
        assert results[0].wins == 2
        assert results[0].win_rate == 1.0
        assert results[0].per_deck_wins == {0: [1.0, 1.0]}

        assert results[1].opponent_label == "model_p_v1"

    def test_agent_reseeded_from_seed_for_each_opponent_match(self, fake_training_env_factory):
        """Episode 1 of opponent A and episode 1 of opponent B both reseed agent to seed+1."""
        FakeEnv, _ = fake_training_env_factory
        agent = _RecordingAgent()
        with patch("yugioh_rl.eval.TrainingEnv", FakeEnv):
            evaluate_with_agent(
                agent,
                deck_pool=[{"main": list(range(40)), "extra": []}],
                opponent_specs=["greedy", "random"],
                num_episodes=2,
                seed=100,
            )

        # 2 opponents × 2 episodes = 4 reseed calls.
        # Each opponent's run_match is called with base_seed=100, so reseeds are
        # 101, 102 (greedy) then 101, 102 (random) — same pair, no drift.
        assert agent.reseed_calls == [101, 102, 101, 102]


# ---------------------------------------------------------------------------
# Parallel-eval primitives — pure-Python tests (no engine required)
# ---------------------------------------------------------------------------


class TestBuildTasks:
    def test_tasks_are_opponent_major(self):
        from yugioh_rl.eval import _build_tasks

        tasks = _build_tasks(["a", "b"], 3)
        # Opponent A's 3 tasks come first, then opponent B's 3.
        assert [(t.opp_idx, t.episode_idx) for t in tasks] == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
        ]

    def test_episodes_are_one_indexed(self):
        from yugioh_rl.eval import _build_tasks

        tasks = _build_tasks(["x"], 3)
        assert [t.episode_idx for t in tasks] == [1, 2, 3]

    def test_zero_episodes_yields_empty(self):
        from yugioh_rl.eval import _build_tasks

        assert _build_tasks(["a", "b"], 0) == []


class TestAggregatePartials:
    def test_groups_by_opp_idx_in_spec_order(self):
        from yugioh_rl.eval import _aggregate_partials, _PartialResult

        partials = [
            _PartialResult(opp_idx=1, episode_idx=1, win=True, agent_deck_idx=0),
            _PartialResult(opp_idx=0, episode_idx=1, win=False, agent_deck_idx=0),
            _PartialResult(opp_idx=0, episode_idx=2, win=True, agent_deck_idx=0),
        ]
        results = _aggregate_partials(partials, ["random", "greedy"])
        assert len(results) == 2
        # Result 0 = "random" (2 episodes, 1 win). Result 1 = "greedy" (1 ep, 1 win).
        assert results[0].opponent_label == "random"
        assert results[0].wins == 1
        assert results[0].episodes == 2
        assert results[1].opponent_label == "greedy"
        assert results[1].wins == 1
        assert results[1].episodes == 1

    def test_per_deck_order_independent_of_reply_order(self):
        """Two reply orders, same final per_deck list ordering — sorted by episode_idx.

        Without the sort, parallel runs at different worker counts could
        produce the same wins/episodes but differently-ordered per_deck
        lists, breaking byte-equal parity assertions in the integration
        test.
        """
        from yugioh_rl.eval import _aggregate_partials, _PartialResult

        # Build 4 partials for one opponent, in episode_idx order: 1,2,3,4.
        # Episodes 1, 3 used deck 0 (win, lose); episodes 2, 4 used deck 1 (lose, win).
        in_order = [
            _PartialResult(0, 1, True, 0),
            _PartialResult(0, 2, False, 1),
            _PartialResult(0, 3, False, 0),
            _PartialResult(0, 4, True, 1),
        ]
        shuffled = [in_order[i] for i in (3, 0, 2, 1)]  # arbitrary worker reply order

        a = _aggregate_partials(in_order, ["x"])
        b = _aggregate_partials(shuffled, ["x"])

        assert (
            a[0].per_deck_wins
            == b[0].per_deck_wins
            == {
                0: [1.0, 0.0],  # deck 0: episode 1 win, episode 3 loss
                1: [0.0, 1.0],  # deck 1: episode 2 loss, episode 4 win
            }
        )

    def test_computes_win_rate(self):
        from yugioh_rl.eval import _aggregate_partials, _PartialResult

        partials = [
            _PartialResult(0, 1, True, 0),
            _PartialResult(0, 2, True, 0),
            _PartialResult(0, 3, False, 0),
            _PartialResult(0, 4, True, 0),
        ]
        results = _aggregate_partials(partials, ["x"])
        assert results[0].wins == 3
        assert results[0].episodes == 4
        assert results[0].win_rate == 0.75

    def test_empty_partials_for_opp(self):
        """An opponent with zero partials still appears with episodes=0, win_rate=0."""
        from yugioh_rl.eval import _aggregate_partials

        results = _aggregate_partials([], ["lonely"])
        assert len(results) == 1
        assert results[0].episodes == 0
        assert results[0].wins == 0
        assert results[0].win_rate == 0.0


# ---------------------------------------------------------------------------
# log_results_to_tensorboard — exact key strings
# ---------------------------------------------------------------------------


class TestLogResultsToTensorboard:
    def test_emits_top_level_and_per_deck_keys(self):
        writer = MagicMock()
        results = [
            EvalResult(
                opponent_label="greedy",
                episodes=10,
                wins=7,
                win_rate=0.7,
                per_deck_wins={0: [1.0, 1.0, 0.0], 1: [0.0, 1.0]},
            ),
        ]
        deck_paths = ["assets/decks/blue_eyes.ydk", "assets/decks/dark_magician.ydk"]

        log_results_to_tensorboard(writer, results, deck_paths, global_step=5000)

        keys = [c[0][0] for c in writer.add_scalar.call_args_list]
        assert "eval/win_rate_vs_greedy" in keys
        assert "eval/win_rate_vs_greedy_deck_blue_eyes" in keys
        assert "eval/win_rate_vs_greedy_deck_dark_magician" in keys

    def test_top_level_value_is_win_rate(self):
        writer = MagicMock()
        results = [
            EvalResult("random", episodes=10, wins=3, win_rate=0.3, per_deck_wins={}),
        ]
        log_results_to_tensorboard(writer, results, deck_paths=[], global_step=42)
        # Look for the "eval/win_rate_vs_random" call, ignore others.
        for c in writer.add_scalar.call_args_list:
            if c[0][0] == "eval/win_rate_vs_random":
                assert c[0][1] == 0.3
                assert c[0][2] == 42
                break
        else:
            pytest.fail("eval/win_rate_vs_random scalar was not written")

    def test_per_deck_value_is_mean(self):
        writer = MagicMock()
        results = [
            EvalResult(
                opponent_label="model_run1_latest",
                episodes=4,
                wins=3,
                win_rate=0.75,
                per_deck_wins={0: [1.0, 1.0, 1.0, 0.0]},
            ),
        ]
        log_results_to_tensorboard(
            writer,
            results,
            deck_paths=["a/b/blue_eyes.ydk"],
            global_step=1,
        )
        per_deck_calls = [
            c
            for c in writer.add_scalar.call_args_list
            if c[0][0] == "eval/win_rate_vs_model_run1_latest_deck_blue_eyes"
        ]
        assert len(per_deck_calls) == 1
        assert per_deck_calls[0][0][1] == pytest.approx(0.75)
