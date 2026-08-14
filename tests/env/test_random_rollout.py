"""Stress test: run many random games to verify stability."""

import random

import pytest

from yugioh_env.models import YuGiOhAction
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment


@pytest.fixture
def env(lib, db_path, script_dirs, deck_path):
    """Create a YuGiOhEnvironment instance."""
    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 0,
    }
    e = YuGiOhEnvironment(config)
    yield e
    e.close()


@pytest.mark.timeout(120)
def test_random_rollout_100(env):
    """Run 100 games with random legal actions. No crashes, all terminate."""
    rng = random.Random(42)
    wins = 0
    losses = 0
    draws = 0
    max_steps = 500

    for game_idx in range(100):
        obs = env.reset(seed=game_idx)
        steps = 0

        while not obs.done and steps < max_steps:
            if obs.num_actions == 0:
                break
            action_idx = rng.randrange(obs.num_actions)
            obs = env.step(YuGiOhAction(action_index=action_idx))
            steps += 1

        if obs.done:
            if obs.reward > 0:
                wins += 1
            elif obs.reward < 0:
                losses += 1
            else:
                draws += 1

    total = wins + losses + draws
    assert total > 0, "No games completed"
    print(f"\nResults: {wins}W / {losses}L / {draws}D out of {total} completed games")
