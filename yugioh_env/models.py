"""OpenEnv Pydantic data models for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)


class YuGiOhAction(Action):
    """Agent action: select an index into available actions."""

    action_index: int = Field(
        ...,
        description=f"Index into available actions (0 to {MAX_ACTIONS - 1})",
        ge=0,
        le=MAX_ACTIONS - 1,
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
      effect:     raw_value=int (effect-desc u64; 0 = no specific effect);  extras={"card_code": int}
                  Emitted only for activate-this-effect actions in
                  MSG_SELECT_IDLECMD / MSG_SELECT_BATTLECMD. Yes/no prompts
                  (MSG_SELECT_YESNO, MSG_SELECT_EFFECTYN) emit no meta —
                  see prompt_meta.desc / prompt_text instead.
    """

    kind: Literal[
        "number",
        "race",
        "attribute",
        "rps",
        "counter",
        "option",
        "chain_link",
        "effect",
    ]
    label: str
    raw_value: int | None = None
    extras: dict = Field(default_factory=dict)


class YuGiOhObservation(Observation):
    """Observation returned to the agent."""

    cards: list[list[int]] = Field(
        default_factory=list,
        description=f"Card features ({MAX_CARDS} x {CARD_FEATURES}) uint8 encoded",
    )
    global_state: list[int] = Field(
        default_factory=list,
        description=f"Global state features ({GLOBAL_FEATURES},) uint8 encoded",
    )
    actions: list[list[int]] = Field(
        default_factory=list,
        description=f"Action features ({MAX_ACTIONS} x {ACTION_FEATURES}) uint8 encoded",
    )
    action_mask: list[int] = Field(
        default_factory=list,
        description=f"Binary action mask ({MAX_ACTIONS},): 1 = legal, 0 = illegal",
    )
    pending_chain: list[list[int]] = Field(
        default_factory=list,
    )
    action_meta: list[ActionMeta | None] = Field(
        default_factory=list,
        description="Per-action prompt metadata, parallel to actions[]; "
        "None for slots without structured meta.",
    )
    prompt_meta: dict | None = Field(
        default=None,
        description="Prompt-level metadata (min/max/cancelable/forced/etc.); "
        "None when no active prompt.",
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
