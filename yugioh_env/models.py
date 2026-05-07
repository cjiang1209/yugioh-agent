"""OpenEnv Pydantic data models for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class YuGiOhAction(Action):
    """Agent action: select an index into available actions."""

    action_index: int = Field(
        ..., description="Index into available actions (0 to 31)", ge=0, le=31
    )


class ActionMeta(BaseModel):
    """Per-action prompt-specific metadata, parallel to YuGiOhObservation.actions[].

    Per-kind contract:
      number:     raw_value=int (the announced number);                   extras={}
      race:       raw_value=int (single-bit RACE_* mask);                 extras={}
      attribute:  raw_value=int (single-bit ATTRIBUTE_* mask);            extras={}
      rps:        raw_value=int (1=Rock, 2=Paper, 3=Scissors);            extras={}
      counter:    raw_value=int (counter_type u16);                       extras={"counter_count": int, "card_code": int}
      option:     raw_value=int (effect-desc u64);                        extras={}
      chain_link: raw_value=int (effect-desc u64);                        extras={"card_code": int}
    """

    kind: Literal[
        "number", "race", "attribute",
        "rps", "counter", "option", "chain_link",
    ]
    label: str
    raw_value: int | None = None
    extras: dict = Field(default_factory=dict)


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
    action_meta: list[ActionMeta | None] = Field(
        default_factory=list,
        description="Per-action prompt metadata, parallel to actions[]; "
                    "None for slots without structured meta.",
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
