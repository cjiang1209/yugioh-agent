"""Env-level per-episode step cap forces a timeout draw."""

from __future__ import annotations

from yugioh_env.models import YuGiOhAction
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

_MAX_STEPS = 5


def _cap_config(db_path, script_dirs, deck_path, max_steps: int) -> dict:
    return {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
        "max_steps": max_steps,
    }


def test_step_cap_forces_timeout_draw(lib, db_path, script_dirs, deck_path):
    """When neither side reaches a terminal state within ``max_steps``
    decisions, the cap force-terminates the episode as a draw.

    A fresh game needs far more than ``max_steps=5`` decisions to finish, so
    the per-player cap must fire (seeded random opponent, agent driven with
    action 0).
    """
    env = YuGiOhEnvironment(_cap_config(db_path, script_dirs, deck_path, _MAX_STEPS))
    try:
        obs = env.reset(seed=42, agent_player=0)
        n = 0
        while not obs.done and n < 200:
            obs = env.step(YuGiOhAction(action_index=0))
            n += 1
        assert obs.done, "episode should terminate"
        assert env._timed_out is True, "termination should be by the step cap"
        assert obs.reward == 0.0, "timeout is a draw (reward 0)"
        assert env._step_count >= _MAX_STEPS or env._opp_step_count >= _MAX_STEPS, (
            "cap is per-player"
        )
    finally:
        env.close()


def _assert_runs_to_natural_end(config: dict) -> None:
    """A disabled cap lets a normal game end naturally, never via timeout."""
    env = YuGiOhEnvironment(config)
    try:
        obs = env.reset(seed=42, agent_player=0)
        steps = 0
        while not obs.done and steps < 5000:
            obs = env.step(YuGiOhAction(action_index=0))
            steps += 1
        assert obs.done
        assert env._timed_out is False
    finally:
        env.close()


def test_step_cap_disabled_by_zero(lib, db_path, script_dirs, deck_path):
    _assert_runs_to_natural_end(_cap_config(db_path, script_dirs, deck_path, 0))


def test_step_cap_disabled_by_negative_value(lib, db_path, script_dirs, deck_path):
    # A negative max_steps must also disable the cap (not just 0/None).
    _assert_runs_to_natural_end(_cap_config(db_path, script_dirs, deck_path, -1))
