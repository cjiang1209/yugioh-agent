"""OpenEnv Pydantic data models for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

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


@dataclass(slots=True, kw_only=True)
class CardRef:
    """A card identified by parsed engine coordinates.

    NEVER construct from filler: `location == 0` is not a valid location
    bitmask and renders as "deck" downstream. Variants for prompts with no
    field card carry a bare `card_code` instead.
    """

    code: int
    controller: int  # relativized: 0=mine, 1=opp
    location: int
    sequence: int


# --- multi-step family (all carry num_selected) ---------------------------
@dataclass(slots=True, kw_only=True)
class PickCard:
    kind: Literal["pick_card"] = "pick_card"
    engine_index: int
    card: CardRef
    num_selected: int
    param: int | None = None  # tribute release_param; sum param (FULL u32)


@dataclass(slots=True, kw_only=True)
class PickBit:
    kind: Literal["pick_bit"] = "pick_bit"
    engine_index: int  # the BIT number
    num_selected: int
    value: int  # the 1<<bit MASK — a different integer


@dataclass(slots=True, kw_only=True)
class FinishPick:
    kind: Literal["finish_pick"] = "finish_pick"
    num_selected: int


# --- single-shot ------------------------------------------------------------
@dataclass(slots=True, kw_only=True)
class CardCommand:
    kind: Literal["card_command"] = "card_command"
    engine_index: int
    command: int  # engine tag t: IDLE_SUMMON..IDLE_SSET
    card: CardRef


@dataclass(slots=True, kw_only=True)
class ActivateEffect:
    kind: Literal["activate_effect"] = "activate_effect"
    engine_index: int
    card: CardRef
    desc: int


@dataclass(slots=True, kw_only=True)
class Attack:
    kind: Literal["attack"] = "attack"
    engine_index: int
    card: CardRef
    direct_attackable: bool


@dataclass(slots=True, kw_only=True)
class PhaseChange:
    kind: Literal["phase_change"] = "phase_change"
    to: Literal["bp", "m2", "ep"]


@dataclass(slots=True, kw_only=True)
class Confirm:
    kind: Literal["confirm"] = "confirm"
    yes: bool
    desc: int


@dataclass(slots=True, kw_only=True)
class ChooseOption:
    kind: Literal["choose_option"] = "choose_option"
    engine_index: int
    desc: int


@dataclass(slots=True, kw_only=True)
class ChoosePosition:
    kind: Literal["choose_position"] = "choose_position"
    position: int  # the position bitmask (was `index`)
    card_code: int


@dataclass(slots=True, kw_only=True)
class PlaceZone:
    kind: Literal["place_zone"] = "place_zone"
    controller: int
    location: int
    sequence: int


@dataclass(slots=True, kw_only=True)
class AnnounceNumber:
    kind: Literal["announce_number"] = "announce_number"
    engine_index: int
    value: int


@dataclass(slots=True, kw_only=True)
class AnnounceCard:
    kind: Literal["announce_card"] = "announce_card"
    card_code: int


@dataclass(slots=True, kw_only=True)
class ChooseRPS:
    kind: Literal["choose_rps"] = "choose_rps"
    choice: int  # 1=Rock 2=Paper 3=Scissors


@dataclass(slots=True, kw_only=True)
class SelectCounter:
    kind: Literal["select_counter"] = "select_counter"
    engine_index: int
    card: CardRef
    counter_type: int  # FULL u16 — flag bits live above 0xFF
    counter_count: int


@dataclass(slots=True, kw_only=True)
class Pass:
    """Chain pass and unselect finish. Fieldless; both submit -1.
    Distinguished from FinishPick, which carries num_selected."""

    kind: Literal["pass"] = "pass"


ActionDescriptor = Annotated[
    PickCard
    | PickBit
    | FinishPick
    | CardCommand
    | ActivateEffect
    | Attack
    | PhaseChange
    | Confirm
    | ChooseOption
    | ChoosePosition
    | PlaceZone
    | AnnounceNumber
    | AnnounceCard
    | ChooseRPS
    | SelectCounter
    | Pass,
    Field(discriminator="kind"),
]


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
    event_history: list[list[int]] = Field(
        default_factory=list,
    )
    action_descriptors: list[ActionDescriptor | None] = Field(
        default_factory=list,
        description="Per-action structured descriptor, parallel to actions[]; "
        "None for padding slots. Empty on terminal observations.",
    )
    prompt_meta: dict | None = Field(
        default=None,
        description="Prompt-level metadata (min/max/cancelable/forced/etc.); "
        "None when no active prompt.",
    )
    events: list[dict] = Field(
        default_factory=list,
        description="Raw enriched engine messages since last action (consumers format them)",
    )


class YuGiOhState(State):
    """Internal environment state metadata."""

    turn_count: int = 0
    phase: str = "draw"
    my_lp: int = 8000
    opp_lp: int = 8000
    my_hand_count: int = 0
    opp_hand_count: int = 0
