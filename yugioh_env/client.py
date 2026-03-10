"""OpenEnv client for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from yugioh_env.models import YuGiOhAction, YuGiOhObservation, YuGiOhState


class YuGiOhEnv(EnvClient[YuGiOhAction, YuGiOhObservation, YuGiOhState]):
    """Client for connecting to a Yu-Gi-Oh! environment server."""

    def reset(
        self,
        *,
        seed: int | None = None,
        deck0: dict[str, list[int]] | None = None,
        deck1: dict[str, list[int]] | None = None,
        agent_player: int | str | None = None,
        **kwargs: Any,
    ) -> StepResult[YuGiOhObservation]:
        """Reset the environment and start a new duel.

        Args:
            seed: RNG seed for this episode.
            deck0: Inline deck for player 0 ({"main": [...], "extra": [...]}).
            deck1: Inline deck for player 1, same format.
            agent_player: Which player the agent controls (0, 1, or "random").
        """
        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if deck0 is not None:
            reset_kwargs["deck0"] = deck0
        if deck1 is not None:
            reset_kwargs["deck1"] = deck1
        if agent_player is not None:
            reset_kwargs["agent_player"] = agent_player
        reset_kwargs.update(kwargs)
        return super().reset(**reset_kwargs)

    def _step_payload(self, action: YuGiOhAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[YuGiOhObservation]:
        obs_data = payload.get("observation", payload)
        reward = payload.get("reward")
        done = payload.get("done", False)
        obs = YuGiOhObservation(**obs_data, reward=reward, done=done)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> YuGiOhState:
        return YuGiOhState(**payload)
