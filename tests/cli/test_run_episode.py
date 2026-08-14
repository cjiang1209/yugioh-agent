"""Tests for cli/play_client.py:run_episode's returned stats.

The stats come off the final observation's `GlobalState`, and `main()` is the
only caller, so nothing else exercises that read. A stub environment drives one
step and ends the duel, which is enough to reach the summary.
"""

from __future__ import annotations

from dataclasses import dataclass

from cli.play_client import run_episode

from yugioh_env.models import GlobalState, Pass, YuGiOhObservation


@dataclass
class _Result:
    observation: YuGiOhObservation
    done: bool
    reward: float


class _OneStepEnv:
    """A live prompt, then the duel ends on the first action."""

    def reset(self, **kwargs) -> _Result:
        obs = YuGiOhObservation(
            action_descriptors=[Pass()],
            global_state=GlobalState(my_lp=8000, opp_lp=7000),
        )
        return _Result(observation=obs, done=False, reward=0.0)

    def step(self, action) -> _Result:
        obs = YuGiOhObservation(global_state=GlobalState(my_lp=8000, opp_lp=0), done=True)
        return _Result(observation=obs, done=True, reward=1.0)


def test_run_episode_reports_the_final_life_points() -> None:
    """The describers are only used for output, so the quiet path can skip
    them; what matters is that the LP are read off the dataclass."""
    stats = run_episode(
        _OneStepEnv(),
        lambda obs: 0,
        action_describer=None,
        event_describer=None,
        verbose=False,
    )
    assert stats == {"steps": 1, "reward": 1.0, "my_lp": 8000, "opp_lp": 0}
