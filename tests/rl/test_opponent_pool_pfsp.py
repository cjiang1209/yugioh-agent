"""Tests for PFSP opponent sampling in OpponentPool."""

from __future__ import annotations

import random as stdlib_random

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from yugioh_rl.opponent_pool import OpponentPool


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def init_hx(self, batch_size: int, device):
        return None


def _network_factory():
    return _Tiny()


def test_sampling_defaults_to_uniform() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        rng=stdlib_random.Random(0),
    )
    assert pool._sampling == "uniform"


def test_sampling_accepts_uniform() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        sampling="uniform",
        rng=stdlib_random.Random(0),
    )
    assert pool._sampling == "uniform"


def test_sampling_accepts_pfsp() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        sampling="pfsp",
        rng=stdlib_random.Random(0),
    )
    assert pool._sampling == "pfsp"


def test_sampling_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="sampling"):
        OpponentPool.create_trainer(
            pool_size=3,
            initial_opponent_spec="random",
            network_factory=_network_factory,
            sampling="random-walk",  # not a valid strategy
            rng=stdlib_random.Random(0),
        )


def _trainer_pool(pool_size: int, sampling: str, seed: int = 0) -> OpponentPool:
    return OpponentPool.create_trainer(
        pool_size=pool_size,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        sampling=sampling,
        rng=stdlib_random.Random(seed),
    )


def _three_slot_pool(
    sampling: str, agent: float, ratings: tuple[float, float, float], seed: int
) -> OpponentPool:
    """Build a 3-occupied-slot pool with the given Elo state."""
    pool = _trainer_pool(pool_size=3, sampling=sampling, seed=seed)
    pool.add_snapshot(_Tiny())  # slot 1
    pool.add_snapshot(_Tiny())  # slot 2
    pool._shared.agent_rating = agent
    for i, r in enumerate(ratings):
        pool._shared.set_rating(i, r)
    return pool


def _sample_counts(pool: OpponentPool, n: int, num_slots: int = 3) -> list[int]:
    counts = [0] * num_slots
    for _ in range(n):
        slot, _ = pool.sample()
        counts[slot] += 1
    return counts


def test_pfsp_single_occupied_slot_returns_zero() -> None:
    pool = _trainer_pool(pool_size=3, sampling="pfsp")
    # Only slot 0 occupied. PFSP should still return 0 deterministically.
    for _ in range(20):
        slot, _ = pool.sample()
        assert slot == 0


def test_pfsp_concentrates_on_closest_to_agent_rating() -> None:
    """With agent at 1500 and slots at [1000, 1500, 2000], hardness weights
    are approximately (0.05, 0.5, 0.95)^2 = (0.0025, 0.25, 0.9025). Slot 2
    should be sampled most often."""
    pool = _three_slot_pool(
        sampling="pfsp", agent=1500.0, ratings=(1000.0, 1500.0, 2000.0), seed=12345
    )
    counts = _sample_counts(pool, n=5000)

    # Slot 2 (hardest) should dominate; slot 0 (easiest) should be smallest.
    assert counts[2] > counts[1] > counts[0], f"unexpected order: {counts}"
    # And slot 2 should clearly beat uniform (~1666).
    assert counts[2] > 2500, f"slot 2 not concentrated enough: {counts}"


def test_pfsp_falls_back_to_uniform_when_agent_dominates_all() -> None:
    """When agent rating >> every slot, win_prob ~ 1 everywhere, weights ~ 0.
    Fallback path samples uniformly so we don't divide by zero or stall."""
    pool = _three_slot_pool(sampling="pfsp", agent=5000.0, ratings=(0.0, 0.0, 0.0), seed=7)
    counts = _sample_counts(pool, n=3000)
    for c in counts:
        assert 800 < c < 1200, f"non-uniform fallback: {counts}"


def test_pfsp_epsilon_mix_engages() -> None:
    """eps=0.2 means roughly 20% of calls take the uniform exploration branch.
    With agent vastly dominant on slots 1,2 but equal to slot 0, pure PFSP
    weighting (no eps) would massively under-sample slots 1 and 2. With the
    eps-mix we expect each of those slots to be sampled at least ~3% of the
    time (eps/3 ~= 6.7%, slack for randomness)."""
    # slot 0 equal => PFSP weight = 0.5^2 = 0.25; slots 1,2 agent dominates => ~0
    pool = _three_slot_pool(sampling="pfsp", agent=1500.0, ratings=(1500.0, 200.0, 200.0), seed=99)
    n = 5000
    counts = _sample_counts(pool, n=n)
    assert counts[1] > 0.03 * n, f"epsilon mix not engaging for slot 1: {counts}"
    assert counts[2] > 0.03 * n, f"epsilon mix not engaging for slot 2: {counts}"


def test_uniform_sampling_unchanged_by_pfsp_code() -> None:
    """Regression: sampling='uniform' must not call any PFSP code paths.
    Set rigged ratings that PFSP would heavily weight, confirm uniform."""
    pool = _three_slot_pool(
        sampling="uniform", agent=1500.0, ratings=(1000.0, 1500.0, 2000.0), seed=0
    )
    counts = _sample_counts(pool, n=3000)
    for c in counts:
        assert 800 < c < 1200, f"uniform broken by PFSP changes: {counts}"


def test_training_env_propagates_sampling_to_pool() -> None:
    """When constructed with opponent_pool_sampling='pfsp', TrainingEnv's
    internal OpponentPool should also have sampling='pfsp'."""
    from pathlib import Path

    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")

    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import TrainingEnv, parse_deck_pool

    config = TrainingConfig(self_play=True, self_play_sampling="pfsp")
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="random",
        network_factory=_network_factory,
        sampling="pfsp",
    )
    deck_pool = parse_deck_pool([str(deck_path), str(deck_path)])

    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_sampling="pfsp",
        opponent_pool_config=config,
        seed=42,
        agent_player="first",
    )
    try:
        assert env._opponent_pool._sampling == "pfsp"
    finally:
        env.close()
