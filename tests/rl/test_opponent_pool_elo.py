"""Tests for Elo tracking in SharedPoolState and OpponentPool."""

from __future__ import annotations

import multiprocessing as mp
import random as stdlib_random

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tests.rl.conftest import requires_engine
from yugioh_rl.opponent_pool import OpponentPool, SharedPoolState


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def init_hx(self, batch_size: int, device):
        return None


def _network_factory():
    return _Tiny()


def test_shared_pool_state_agent_rating_default_is_1500() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    assert state.agent_rating == pytest.approx(1500.0)


def test_shared_pool_state_ratings_default_is_1500_per_slot() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    for i in range(3):
        assert state.get_rating(i) == pytest.approx(1500.0)


def test_shared_pool_state_n_games_starts_zero_per_slot() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    for i in range(3):
        assert state.get_n_games(i) == 0


def test_shared_pool_state_agent_rating_round_trips() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.agent_rating = 1730.5
    assert state.agent_rating == pytest.approx(1730.5)


def test_shared_pool_state_set_rating_round_trips() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.set_rating(1, 1612.25)
    assert state.get_rating(1) == pytest.approx(1612.25)
    # Other slots untouched.
    assert state.get_rating(0) == pytest.approx(1500.0)
    assert state.get_rating(2) == pytest.approx(1500.0)


def test_shared_pool_state_increment_n_games() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.increment_n_games(2)
    state.increment_n_games(2)
    assert state.get_n_games(2) == 2
    assert state.get_n_games(0) == 0


def _child_read_elo(handles, send_pipe):
    from yugioh_rl.opponent_pool import SharedPoolState

    state = SharedPoolState.from_handles(handles)
    send_pipe.send((state.agent_rating, state.get_rating(1), state.get_n_games(1)))
    send_pipe.close()


def test_shared_pool_state_cross_process_elo_visibility() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.agent_rating = 1611.0
    state.set_rating(1, 1422.5)
    state.increment_n_games(1)
    state.increment_n_games(1)
    state.increment_n_games(1)

    ctx = mp.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe()
    proc = ctx.Process(target=_child_read_elo, args=(state.share_handles(), child_pipe))
    proc.start()
    child_pipe.close()
    proc.join(timeout=30)
    assert proc.exitcode == 0

    agent, slot1, ngames = parent_pipe.recv()
    assert agent == pytest.approx(1611.0)
    assert slot1 == pytest.approx(1422.5)
    assert ngames == 3


def test_sample_returns_slot_and_opponent_tuple() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    result = pool.sample()
    assert isinstance(result, tuple)
    assert len(result) == 2
    slot, opp = result
    assert slot == 0  # only slot 0 occupied
    assert opp is not None


def test_report_result_win_increases_agent_rating() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    before = pool._shared.agent_rating
    pool.report_result(slot=0, agent_won=True)
    after = pool._shared.agent_rating
    assert after > before


def test_report_result_loss_decreases_agent_rating() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    before = pool._shared.agent_rating
    pool.report_result(slot=0, agent_won=False)
    after = pool._shared.agent_rating
    assert after < before


def test_report_result_is_zero_sum() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    total_before = pool._shared.agent_rating + pool._shared.get_rating(0)
    pool.report_result(slot=0, agent_won=True)
    total_after = pool._shared.agent_rating + pool._shared.get_rating(0)
    assert total_after == pytest.approx(total_before)


def test_report_result_increments_n_games() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    pool.report_result(slot=0, agent_won=True)
    pool.report_result(slot=0, agent_won=False)
    assert pool._shared.get_n_games(0) == 2


def test_add_snapshot_inherits_agent_rating() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    pool._shared.agent_rating = 1742.0
    # Play one game to give slot 0 some n_games history.
    pool.report_result(slot=0, agent_won=True)
    agent_rating_after_game = pool._shared.agent_rating

    net = _Tiny()
    new_slot = pool.add_snapshot(net)

    # New snapshot lands in slot 1 (slot 0 already occupied by initial).
    assert new_slot == 1
    assert pool._shared.get_rating(new_slot) == pytest.approx(agent_rating_after_game)
    assert pool._shared.get_n_games(new_slot) == 0


def test_add_snapshot_resets_n_games_when_overwriting_slot() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    # Fill: slot 0 (init), then slot 1 (snapshot 1), then wrap to slot 0
    # (snapshot 2 overwrites init).
    pool.add_snapshot(_Tiny())  # slot 1
    pool.report_result(slot=0, agent_won=True)
    pool.report_result(slot=0, agent_won=True)
    assert pool._shared.get_n_games(0) == 2

    pool.add_snapshot(_Tiny())  # wraps, overwrites slot 0
    assert pool._shared.get_n_games(0) == 0


def test_elo_summary_reports_occupied_slots_only() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=4,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    pool._shared.agent_rating = 1600.0
    pool._shared.set_rating(0, 1500.0)
    # Only slot 0 occupied; slots 1-3 unoccupied.
    summary = pool.elo_summary()
    assert summary["agent"] == pytest.approx(1600.0)
    assert summary["pool_mean"] == pytest.approx(1500.0)
    assert summary["pool_min"] == pytest.approx(1500.0)
    assert summary["pool_max"] == pytest.approx(1500.0)
    assert summary["occupied"] == 1


def test_elo_summary_aggregates_multiple_slots() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    pool.add_snapshot(_Tiny())
    pool.add_snapshot(_Tiny())
    pool._shared.set_rating(0, 1400.0)
    pool._shared.set_rating(1, 1500.0)
    pool._shared.set_rating(2, 1600.0)
    summary = pool.elo_summary()
    assert summary["occupied"] == 3
    assert summary["pool_mean"] == pytest.approx(1500.0)
    assert summary["pool_min"] == pytest.approx(1400.0)
    assert summary["pool_max"] == pytest.approx(1600.0)


@requires_engine
def test_self_play_pool_elo_updates_during_real_episode() -> None:
    """End-to-end: drive a real TrainingEnv attached to an OpponentPool through
    at least one episode, and assert agent_rating moved and n_games incremented."""
    from pathlib import Path

    import numpy as np

    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")

    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import TrainingEnv, parse_deck_pool
    from yugioh_rl.network import YuGiOhNet
    from yugioh_rl.opponent_pool import OpponentPool

    config = TrainingConfig(self_play=True)
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="random",
        network_factory=lambda: YuGiOhNet.from_config(config),
    )

    deck_pool = parse_deck_pool([str(deck_path), str(deck_path)])

    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_config=config,
        reward_shaping=False,
        seed=42,
        agent_player="first",
    )

    initial_agent = pool._shared.agent_rating
    initial_n_games_total = sum(pool._shared.get_n_games(i) for i in range(2))

    try:
        obs = env.reset()
        finished_episodes = 0
        steps = 0
        # Run up to 2 episodes (or 1600 steps) — usually one is enough for
        # the assertions, two gives a margin.
        while finished_episodes < 2 and steps < 1600:
            action = int(np.argmax(obs["action_mask"]))
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                finished_episodes += 1
                if finished_episodes < 2:
                    obs = env.reset()
        if finished_episodes == 0:
            pytest.skip(f"no episode completed within {steps} steps")
    finally:
        env.close()

    final_agent = pool._shared.agent_rating
    final_n_games_total = sum(pool._shared.get_n_games(i) for i in range(2))

    assert final_n_games_total > initial_n_games_total, (
        f"expected n_games to increment, got {initial_n_games_total} → {final_n_games_total}"
    )
    assert final_agent != initial_agent, (
        f"expected agent_rating to move, got {initial_agent} → {final_agent}"
    )


def test_elo_summary_dict_shape_pinned_for_ppo_logging() -> None:
    """PPO logging code expects these exact keys; if you rename one, you
    must also update yugioh_rl/ppo.py — this test is the canary."""
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    summary = pool.elo_summary()
    assert set(summary.keys()) == {"agent", "pool_mean", "pool_min", "pool_max", "occupied"}
