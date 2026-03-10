"""OpenEnv Pydantic data models for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class YuGiOhAction(Action):
    """Agent action: select an index into available actions."""

    action_index: int = Field(
        ..., description="Index into available actions (0 to 31)", ge=0, le=31
    )


class YuGiOhObservation(Observation):
    """Observation returned to the agent."""

    cards: list[list[int]] = Field(
        default_factory=list,
        description="Card features (200 x 42) uint8 encoded",
    )
    global_state: list[int] = Field(
        default_factory=list,
        description="Global state features (20,) uint8 encoded",
    )
    actions: list[list[int]] = Field(
        default_factory=list,
        description="Action features (32 x 12) uint8 encoded",
    )
    action_mask: list[int] = Field(
        default_factory=list,
        description="Binary action mask (32,): 1 = legal, 0 = illegal",
    )
    event_log: list[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of events since last action",
    )


class YuGiOhState(State):
    """Internal environment state metadata."""

    turn_count: int = 0
    phase: str = "draw"
    my_lp: int = 8000
    opp_lp: int = 8000
    my_hand_count: int = 0
    opp_hand_count: int = 0
