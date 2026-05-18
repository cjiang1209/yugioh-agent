"""Test YuGiOhEnvironment.set_opponent for opponent swapping between episodes."""
from __future__ import annotations

from yugioh_env.opponent import GreedyOpponent, RandomOpponent


def test_set_opponent_replaces_instance(lib, db_path, script_dirs) -> None:
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment(config={"opponent": "greedy"})
    try:
        assert isinstance(env._opponent, GreedyOpponent)
        env.set_opponent(RandomOpponent(seed=0))
        assert isinstance(env._opponent, RandomOpponent)
    finally:
        env.close()
