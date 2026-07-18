"""Tests for yugioh_rl.eval — core behavior of the standalone eval module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.rl.conftest import requires_engine
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
    _aggregate_one,
    _EpisodeRecord,
    eval_result_to_row,
    evaluate_with_agent,
    make_eval_agent,
    opponent_label_from_spec,
    run_match,
)

_DECK_PATH = Path("assets/decks/blue_eyes.ydk")


def _deck_pool_or_skip():
    from yugioh_rl.env_wrapper import parse_deck_pool

    if not _DECK_PATH.exists():
        pytest.skip(f"missing deck: {_DECK_PATH}")
    return parse_deck_pool([str(_DECK_PATH)])


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
            info["episode_length"] = outcome.get("episode_length", 0)
            info["turn_count"] = outcome.get("turn_count", 0)
            info["agent_player"] = outcome.get("agent_player", 0)
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
        records = run_match(agent, env, num_episodes=4, base_seed=42)
        wins = sum(1 for r in records if r.win)
        per_deck: dict[int, list[float]] = {}
        for r in records:
            per_deck.setdefault(r.agent_deck_idx, []).append(1.0 if r.win else 0.0)
        assert wins == 3
        assert per_deck == {0: [1.0, 0.0], 1: [1.0, 1.0]}

    def test_returns_episode_records_with_steps_turns_and_order(self):
        """New fields flow end-to-end: terminal info -> _EpisodeRecord."""
        agent = _RecordingAgent()
        env = _ScriptedEnv(
            [
                [
                    {
                        "done": True,
                        "reward": 1.0,
                        "agent_deck_idx": 0,
                        "episode_length": 3,
                        "turn_count": 2,
                        "agent_player": 0,
                    }
                ],
                [
                    {
                        "done": True,
                        "reward": -1.0,
                        "agent_deck_idx": 0,
                        "episode_length": 5,
                        "turn_count": 3,
                        "agent_player": 1,
                    }
                ],
                [
                    {
                        "done": True,
                        "reward": 1.0,
                        "agent_deck_idx": 1,
                        "episode_length": 4,
                        "turn_count": 2,
                        "agent_player": 0,
                    }
                ],
                [
                    {
                        "done": True,
                        "reward": 1.0,
                        "agent_deck_idx": 1,
                        "episode_length": 6,
                        "turn_count": 4,
                        "agent_player": 1,
                    }
                ],
            ]
        )
        recs = run_match(agent, env, num_episodes=4, base_seed=42)
        assert all(isinstance(r, _EpisodeRecord) for r in recs)
        assert [r.steps for r in recs] == [3, 5, 4, 6]
        assert [r.turns for r in recs] == [2, 3, 2, 4]
        assert [r.went_first for r in recs] == [True, False, True, False]
        assert [r.episode_idx for r in recs] == [1, 2, 3, 4]

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
        records = run_match(agent, env, num_episodes=0, base_seed=42)
        assert records == []
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
        with patch("yugioh_rl.eval.EvalEnv", FakeEnv):
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
            assert env.kwargs["deck_allocation"] == "random"
            assert env.kwargs["mirror_decks"] is False
            assert env.kwargs["seed"] == 42
            assert env.kwargs["agent_player"] == "random"
            # opponent_device omitted → kwarg absent so env-var fallback wins.
            assert "opponent_device" not in env.kwargs

    def test_opponent_device_forwarded_only_when_provided(self, fake_training_env_factory):
        FakeEnv, instances = fake_training_env_factory
        agent = _RecordingAgent()
        with patch("yugioh_rl.eval.EvalEnv", FakeEnv):
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
        with patch("yugioh_rl.eval.EvalEnv", FakeEnv):
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
        with patch("yugioh_rl.eval.EvalEnv", FakeEnv):
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
            _PartialResult(
                opp_idx=1,
                episode_idx=1,
                win=True,
                agent_deck_idx=0,
                steps=1,
                turns=1,
                went_first=True,
            ),
            _PartialResult(
                opp_idx=0,
                episode_idx=1,
                win=False,
                agent_deck_idx=0,
                steps=1,
                turns=1,
                went_first=True,
            ),
            _PartialResult(
                opp_idx=0,
                episode_idx=2,
                win=True,
                agent_deck_idx=0,
                steps=1,
                turns=1,
                went_first=True,
            ),
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
            _PartialResult(0, 1, True, 0, 1, 1, True),
            _PartialResult(0, 2, False, 1, 1, 1, True),
            _PartialResult(0, 3, False, 0, 1, 1, True),
            _PartialResult(0, 4, True, 1, 1, 1, True),
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
            _PartialResult(0, 1, True, 0, 1, 1, True),
            _PartialResult(0, 2, True, 0, 1, 1, True),
            _PartialResult(0, 3, False, 0, 1, 1, True),
            _PartialResult(0, 4, True, 0, 1, 1, True),
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
# _aggregate_one — the shared aggregator behind both the sequential and
# parallel (_aggregate_partials) paths.
# ---------------------------------------------------------------------------


class TestAggregateOne:
    def test_computes_all_metrics(self):
        recs = [
            _EpisodeRecord(1, True, 0, 10, 4, True),
            _EpisodeRecord(2, False, 0, 20, 6, False),
            _EpisodeRecord(3, True, 1, 30, 8, True),
            _EpisodeRecord(4, True, 1, 40, 10, False),
        ]
        r = _aggregate_one(recs, "opp")
        assert (r.episodes, r.wins, r.win_rate) == (4, 3, 0.75)
        assert (r.steps_mean, r.steps_median, r.steps_max) == (25.0, 25.0, 40)
        assert (r.turns_mean, r.turns_max) == (7.0, 10)
        assert (r.wins_first, r.episodes_first) == (2, 2)
        assert (r.wins_second, r.episodes_second) == (1, 2)
        assert r.per_deck_wins == {0: [1.0, 0.0], 1: [1.0, 1.0]}

    def test_empty_and_singleton(self):
        assert _aggregate_one([], "opp").episodes == 0
        one = _aggregate_one([_EpisodeRecord(1, True, 0, 5, 3, True)], "opp")
        assert one.steps_std == 0.0 and one.steps_mean == 5.0


# ---------------------------------------------------------------------------
# eval_result_to_row — new metrics carried through to the JSON-able row.
# ---------------------------------------------------------------------------


class TestEvalResultToRow:
    def test_carries_new_fields(self):
        r = EvalResult(
            opponent_label="opp",
            episodes=4,
            wins=3,
            per_deck_wins={0: [1.0, 0.0]},
            steps_mean=25.0,
            steps_std=1.0,
            steps_median=25.0,
            steps_max=40,
            turns_mean=7.0,
            turns_std=2.0,
            turns_median=7.0,
            turns_max=10,
            wins_first=2,
            episodes_first=2,
            wins_second=1,
            episodes_second=2,
        )
        row = eval_result_to_row(r, ["blue_eyes"])
        assert row["steps"] == {"mean": 25.0, "std": 1.0, "median": 25.0, "max": 40}
        # play_first_rate is derived from the order-split counts (2 of 4 episodes first).
        assert row["turns"]["mean"] == 7.0 and row["play_first_rate"] == 0.5
        assert row["episodes_first"] == 2 and row["wins_first"] == 2


# ---------------------------------------------------------------------------
# Engine-gated: real EvalEnv terminal info + parallel/sequential parity for
# the new per-episode metrics. Skip (via requires_engine) when libocgcore /
# cards.cdb are absent.
# ---------------------------------------------------------------------------


@requires_engine
def test_evalenv_terminal_info_has_turn_and_player() -> None:
    from yugioh_rl.env_wrapper import EvalEnv

    deck_pool = _deck_pool_or_skip()
    env = EvalEnv(deck_pool=deck_pool, opponent="random", seed=0, agent_player="first")
    try:
        env.reset(episode_idx=1)
        info, done = {}, False
        while not done:
            _obs, _reward, done, info = env.step(0)
        assert info["agent_player"] == 0
        assert isinstance(info["turn_count"], int) and info["turn_count"] >= 1
        assert "episode_length" in info
    finally:
        env.close()


@requires_engine
def test_parallel_matches_sequential_new_fields() -> None:
    from yugioh_rl.eval import evaluate

    kw = dict(
        deck_pool=_deck_pool_or_skip(),
        opponent_specs=["random"],
        num_episodes=6,
        seed=0,
        agent_player="random",
    )
    r1 = evaluate("random", workers=1, **kw)[0]
    r2 = evaluate("random", workers=2, **kw)[0]
    for f in (
        "wins",
        "steps_mean",
        "steps_std",
        "turns_mean",
        "wins_first",
        "episodes_first",
        "wins_second",
        "episodes_second",
    ):
        assert getattr(r1, f) == getattr(r2, f), f
