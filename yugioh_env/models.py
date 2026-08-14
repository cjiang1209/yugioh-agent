"""OpenEnv Pydantic data models for the Yu-Gi-Oh! environment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Literal

import numpy as np
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BeforeValidator, ConfigDict, Field, PlainSerializer, WithJsonSchema

from yugioh_core.encoding import (
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    MAX_ACTIONS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
)


def _to_array(v, *, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Normalize ANY input to exactly `shape`/`dtype`.

    ndarrays need normalizing too, not just lists: `np.array([], np.uint8)`
    arrives as shape (0,) and would otherwise pass straight through. A 1-D
    input is reshaped; a multi-dim one must already match, since reshape
    would silently transpose a same-size mismatch rather than raise.

    No copy: a right-shape, right-dtype ndarray reshapes to a view, so the
    field ALIASES the producer's buffer. Every producer must therefore return
    a freshly allocated array -- a reused buffer corrupts readers with no
    error, e.g. `yugioh_rl`'s actor-learner holds one `Transition` per step
    and copies only at the `np.stack` packing the rollout, so every step
    would collapse onto the buffer's last contents.
    """
    if v is None:
        return np.zeros(shape, dtype)
    try:
        arr = np.asarray(v, dtype=dtype)
    except OverflowError as e:
        # pydantic wraps ValueError into ValidationError but not OverflowError,
        # so an out-of-range int would otherwise escape raw to the caller.
        raise ValueError(f"value out of range for {np.dtype(dtype).name}: {e}") from e
    if arr.size == 0:
        return np.zeros(shape, dtype)
    if arr.ndim > 1 and arr.shape != shape:
        raise ValueError(
            f"array shape {arr.shape} is not compatible with field shape {shape} "
            "(same element count but transposed/misshapen dims -- refusing to "
            "silently reshape multi-dimensional input)"
        )
    return arr.reshape(shape)


def _int_array_schema(ndim: int) -> dict:
    s: dict = {"type": "integer"}
    for _ in range(ndim):
        s = {"type": "array", "items": s}
    return s


def NDArrayField(shape: tuple[int, ...], dtype=np.uint8):
    return Annotated[
        np.ndarray,
        BeforeValidator(partial(_to_array, shape=shape, dtype=dtype)),
        PlainSerializer(lambda a: a.tolist(), return_type=list),
        WithJsonSchema(_int_array_schema(len(shape))),
        Field(default_factory=lambda: np.zeros(shape, dtype)),
    ]


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
    # 4th loc_info slot, overloaded by the engine: the material's index in the
    # overlay stack when location & LOCATION_OVERLAY, else the position bitmask.
    subsequence: int = 0


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
    # The activating card's position bitmask, as the chain extractor reports it.
    position: int = 0


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


@dataclass(slots=True, kw_only=True)
class CardState:
    """One card as the engine reported it.

    Fields mirror ``encode_card``'s parameters so packing is mechanical. Values
    are raw: clamping and masking happen only when encoding to bytes.

    A dataclass rather than a model: one is built per card on every board
    observation, and the values come straight from the engine, so validation
    cost buys nothing here.

    A hidden card is a real card with ``code=0`` and ``is_public=False`` --
    its zone and seat are known, only its identity is withheld.
    """

    code: int = 0
    location: int = 0
    sequence: int = 0
    position: int = 0
    controller: int = 0
    is_public: bool = False
    card_type: int = 0
    level: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    lscale: int = 0
    rscale: int = 0
    link_marker: int = 0
    counter_count: int = 0
    negated: bool = False
    is_overlay: bool = False


@dataclass(slots=True, kw_only=True)
class GlobalState:
    """Duel-level scalars as the engine reported them, agent-relative.

    A dataclass rather than a model, for the same reason as `CardState`.

    ``phase`` is the engine bitmask, not a name: both the encoder and the
    ygo-agent bridge consume the bitmask, and naming is a rendering concern.
    """

    my_lp: int = 0
    opp_lp: int = 0
    turn: int = 0
    phase: int = 0
    is_my_turn: bool = False
    chain_count: int = 0
    msg_type: int = 0
    my_deck: int = 0
    my_hand: int = 0
    my_grave: int = 0
    my_banished: int = 0
    my_extra: int = 0
    opp_deck: int = 0
    opp_hand: int = 0
    opp_grave: int = 0
    opp_banished: int = 0
    opp_extra: int = 0
    is_finished: bool = False


class YuGiOhObservation(Observation):
    """Observation returned to the agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pending_chain: NDArrayField((MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)) = Field(
        description=f"Pending chain features ({MAX_PENDING_CHAIN} x {CHAIN_ENTRY_FEATURES}) uint8 encoded",
    )
    event_history: NDArrayField((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)) = Field(
        description=f"Event history features ({MAX_EVENT_HISTORY} x {EVENT_ENTRY_FEATURES}) uint8 encoded",
    )
    action_descriptors: list[ActionDescriptor] = Field(
        default_factory=list,
        description="One structured descriptor per legal action, in engine "
        "order. Empty on terminal observations.",
    )
    cards: list[CardState] = Field(
        default_factory=list,
        description="Structured board cards, raw engine values, in engine query order. "
        "Hidden and known cards interleave, and the network's row ordering follows "
        "this order, so consumers must not filter or reorder it.",
    )
    global_state: GlobalState = Field(
        default_factory=GlobalState,
        description="Structured duel scalars, raw engine values",
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

    @property
    def num_actions(self) -> int:
        """Number of legal actions for the active prompt."""
        return len(self.action_descriptors)

    @property
    def msg_type(self) -> int:
        """Engine message type of the active prompt, or 0 when there is none.

        Spares readers the ``prompt_meta`` lookup and its None guard.
        """
        return self.prompt_meta.get("msg_type", 0) if self.prompt_meta else 0


class YuGiOhState(State):
    """Internal environment state metadata."""

    turn_count: int = 0
    phase: str = "draw"
    my_lp: int = 8000
    opp_lp: int = 8000
    my_hand_count: int = 0
    opp_hand_count: int = 0
