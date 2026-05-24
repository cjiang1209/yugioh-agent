"""Tests for TrainingEnv self-play integration."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yugioh_env.server.yugioh_environment")

from tests.rl.conftest import requires_engine
from yugioh_rl.env_wrapper import parse_deck_pool


def _make_deck_pool() -> list[dict[str, list[int]]]:
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    return parse_deck_pool([str(deck_path)])


@requires_engine
def test_training_env_swaps_opponent_per_episode() -> None:
    from yugioh_env.opponent import RandomOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import TrainingEnv
    from yugioh_rl.network import YuGiOhNet
    from yugioh_rl.opponent_pool import OpponentPool

    config = TrainingConfig(self_play=True)
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="random",
        network_factory=lambda: YuGiOhNet.from_config(config),
    )

    env = TrainingEnv(
        deck_pool=_make_deck_pool(),
        opponent="random",
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_config=config,
        seed=42,
    )
    try:
        env.reset()
        assert isinstance(env._env._opponent, RandomOpponent)
    finally:
        env.close()


@requires_engine
def test_training_env_snapshot_opponent_uses_yugioh_net() -> None:
    """With pool_size=1, the first add_snapshot evicts the scripted seed,
    so sample() deterministically returns the snapshot NetworkOpponent."""
    from yugioh_env.opponent import NetworkOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import TrainingEnv
    from yugioh_rl.network import YuGiOhNet
    from yugioh_rl.opponent_pool import OpponentPool

    config = TrainingConfig(self_play=True)
    pool = OpponentPool.create_trainer(
        pool_size=1,
        initial_opponent_spec="greedy",
        network_factory=lambda: YuGiOhNet.from_config(config),
    )
    pool.add_snapshot(YuGiOhNet.from_config(config))

    env = TrainingEnv(
        deck_pool=_make_deck_pool(),
        opponent="greedy",
        opponent_pool_handles=pool.share_handles(),
        opponent_pool_temperature=1.0,
        opponent_pool_config=config,
        seed=42,
    )
    try:
        env.reset()
        assert isinstance(env._env._opponent, NetworkOpponent)
    finally:
        env.close()
