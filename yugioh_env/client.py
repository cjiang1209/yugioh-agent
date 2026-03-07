"""OpenEnv client for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from yugioh_env.models import YuGiOhAction, YuGiOhObservation, YuGiOhState


class YuGiOhEnv(EnvClient[YuGiOhAction, YuGiOhObservation, YuGiOhState]):
    """Client for connecting to a Yu-Gi-Oh! environment server."""

    def _step_payload(self, action: YuGiOhAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[YuGiOhObservation]:
        obs = YuGiOhObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict) -> YuGiOhState:
        return YuGiOhState(**payload)
