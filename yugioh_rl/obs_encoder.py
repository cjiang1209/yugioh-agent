"""Pack a structured observation into the uint8 arrays the network reads.

Feature encoding is network-specific, so it lives here rather than on the
observation, which stays a data carrier. Called by ``TrainingEnv`` for
collection and ``NetworkOpponent`` for inference.

Every clamp and mask in this module exists to fit a uint8 array: the structured
models hold raw engine values, and the fields packed here are narrowed here.
``pending_chain`` and ``event_history`` arrive already packed and pass through.
"""

from __future__ import annotations

import numpy as np

from yugioh_core.action_categories import (
    BATTLE_ACTIVATE,
    BATTLE_ATTACK,
    BATTLE_TO_EP,
    BATTLE_TO_M2,
    IDLE_ACTIVATE,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from yugioh_core.constants import (
    LOCATION_MZONE,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
)
from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    encode_global,
    pack_action_into,
    pack_card_into,
)
from yugioh_env.models import (
    ActivateEffect,
    AnnounceCard,
    AnnounceNumber,
    Attack,
    CardCommand,
    ChooseOption,
    ChoosePosition,
    ChooseRPS,
    Confirm,
    FinishPick,
    Pass,
    PhaseChange,
    PickBit,
    PickCard,
    PlaceZone,
    SelectCounter,
    YuGiOhObservation,
)


def _encode_cards(obs: YuGiOhObservation) -> np.ndarray:
    """Pack the board into one buffer, a row per card.

    Every card writes straight into the shared buffer, so a board costs one
    allocation. Unoccupied rows keep the zeros the buffer starts with.
    """
    buf = bytearray(MAX_CARDS * CARD_FEATURES)
    for i, c in enumerate(obs.cards[:MAX_CARDS]):
        # Positional, and the order is the layout's: keyword arguments cost
        # about as much again as the packing itself at this call rate.
        pack_card_into(
            buf,
            i * CARD_FEATURES,
            c.code,
            c.location,
            c.sequence,
            c.position,
            c.controller,
            c.is_public,
            c.card_type,
            c.level,
            c.attribute,
            c.race,
            c.attack,
            c.defense,
            c.lscale,
            c.rscale,
            c.link_marker,
            c.counter_count,
            c.negated,
            c.is_overlay,
        )
    return np.frombuffer(buf, dtype=np.uint8).reshape(MAX_CARDS, CARD_FEATURES)


def _encode_global(obs: YuGiOhObservation) -> np.ndarray:
    s = obs.global_state
    return encode_global(
        s.my_lp,
        s.opp_lp,
        s.turn,
        s.phase,
        s.is_my_turn,
        s.chain_count,
        s.msg_type,
        s.my_deck,
        s.my_hand,
        s.my_grave,
        s.my_banished,
        s.my_extra,
        s.opp_deck,
        s.opp_hand,
        s.opp_grave,
        s.opp_banished,
        s.opp_extra,
        s.is_finished,
    )


def _encode_action_mask(obs: YuGiOhObservation) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[: len(obs.action_descriptors)] = 1
    return mask


def _activate_category(msg_type: int) -> int:
    """`activate_effect` carries no category of its own -- idle, battle and
    chain each need a different byte value, and msg_type is the only thing
    on the descriptor that tells them apart. Those three prompts are the only
    ones that offer an activation, so any other means the descriptor and the
    prompt disagree -- guessing would pick a wrong byte in silence."""
    if msg_type == MSG_SELECT_IDLECMD:
        return IDLE_ACTIVATE
    if msg_type == MSG_SELECT_BATTLECMD:
        return BATTLE_ACTIVATE
    if msg_type == MSG_SELECT_CHAIN:
        return 0  # a chain activation carries no category
    raise ValueError(f"effect activation under msg_type {msg_type}")


def _phase_change_category(msg_type: int, to: str) -> int:
    """Same story as `_activate_category`: `to` plus msg_type is what
    distinguishes the four phase-change byte values. Only the idle and battle
    prompts offer a phase change, so any other msg_type means the descriptor
    and the prompt disagree -- guessing would pick a wrong byte in silence."""
    if msg_type == MSG_SELECT_IDLECMD:
        return IDLE_TO_BP if to == "bp" else IDLE_TO_EP
    if msg_type == MSG_SELECT_BATTLECMD:
        return BATTLE_TO_M2 if to == "m2" else BATTLE_TO_EP
    raise ValueError(f"phase change to {to!r} under msg_type {msg_type}")


def _confirm_card_fields(obs: YuGiOhObservation) -> tuple[int, int, int, int]:
    """`Confirm` only carries `yes`/`desc`. The confirmed card's code,
    controller, location and sequence live on `obs.prompt_meta`, populated
    by `_build_prompt_meta` for `MSG_SELECT_EFFECTYN`. `MSG_SELECT_YESNO`
    never carries a card, so zero is correct for every other msg_type too.
    """
    if obs.msg_type != MSG_SELECT_EFFECTYN or obs.prompt_meta is None:
        return 0, 0, 0, 0
    pm = obs.prompt_meta
    return (
        pm.get("card_code", 0),
        pm.get("controller", 0),
        pm.get("location", 0),
        pm.get("sequence", 0),
    )


def _row_fields(d, i: int, msg_type: int, obs: YuGiOhObservation) -> dict:
    """Reconstruct the raw values the action layout needs, one branch per
    descriptor kind. A kind that carries no value for a field leaves the
    default below, which the frozen goldens hold it to.
    """
    f = {
        "category": 0,
        "code": 0,
        "controller": 0,
        "location": 0,
        "sequence": 0,
        "subsequence": 0,
        "position": 0,
        "direct_attackable": 0,
        "param": 0,
        "counter_type": 0,
        "counter_count": 0,
        "index": 0,
        "num_selected": 1,
        "desc": 0,
    }
    if isinstance(d, PickCard):
        f["code"], f["controller"] = d.card.code, d.card.controller
        f["location"], f["sequence"] = d.card.location, d.card.sequence
        f["subsequence"] = d.subsequence
        f["param"] = d.param or 0
        f["index"] = d.engine_index
        f["num_selected"] = d.num_selected
    elif isinstance(d, PickBit):
        f["index"] = d.engine_index
        f["num_selected"] = d.num_selected
    elif isinstance(d, FinishPick):
        f["category"] = 1
        f["num_selected"] = d.num_selected
    elif isinstance(d, CardCommand):
        f["category"] = d.command
        f["code"], f["controller"] = d.card.code, d.card.controller
        f["location"], f["sequence"] = d.card.location, d.card.sequence
        f["index"] = d.engine_index
    elif isinstance(d, ActivateEffect):
        f["category"] = _activate_category(msg_type)
        f["code"], f["controller"] = d.card.code, d.card.controller
        f["location"], f["sequence"] = d.card.location, d.card.sequence
        f["position"] = d.position
        f["desc"] = d.desc
        f["index"] = d.engine_index
    elif isinstance(d, Attack):
        f["category"] = BATTLE_ATTACK
        f["code"], f["controller"] = d.card.code, d.card.controller
        f["location"], f["sequence"] = d.card.location, d.card.sequence
        f["direct_attackable"] = 1 if d.direct_attackable else 0
        f["index"] = d.engine_index
    elif isinstance(d, PhaseChange):
        f["category"] = _phase_change_category(msg_type, d.to)
    elif isinstance(d, Confirm):
        f["category"] = 0 if d.yes else 1
        f["desc"] = d.desc
        f["code"], f["controller"], f["location"], f["sequence"] = _confirm_card_fields(obs)
    elif isinstance(d, ChooseOption):
        f["desc"] = d.desc
        f["index"] = d.engine_index
    elif isinstance(d, ChoosePosition):
        f["code"] = d.card_code
        f["index"] = d.position
    elif isinstance(d, PlaceZone):
        f["category"] = 0 if d.location == LOCATION_MZONE else 1
        f["controller"], f["location"], f["sequence"] = d.controller, d.location, d.sequence
        # PlaceZone has no engine_index-shaped field. `place_zone` is the
        # only descriptor kind its msg_type ever produces, so the row's own
        # position in the array is itself a valid sequential index.
        f["index"] = i
    elif isinstance(d, AnnounceNumber):
        f["index"] = d.engine_index
    elif isinstance(d, AnnounceCard):
        f["code"] = d.card_code
        # Same reasoning as PlaceZone: no stored index, but announce_card is
        # the only descriptor kind MSG_ANNOUNCE_CARD produces, so the row's
        # own position is a valid sequential index here too.
        f["index"] = i
    elif isinstance(d, ChooseRPS):
        f["index"] = d.choice
    elif isinstance(d, SelectCounter):
        f["code"], f["controller"] = d.card.code, d.card.controller
        f["location"], f["sequence"] = d.card.location, d.card.sequence
        f["counter_type"], f["counter_count"] = d.counter_type, d.counter_count
        f["index"] = d.engine_index
    elif isinstance(d, Pass):
        f["category"] = 1
    else:
        raise ValueError(f"unhandled action descriptor: {d!r}")
    return f


def _encode_actions(obs: YuGiOhObservation) -> np.ndarray:
    """Pack the prompt's actions into one buffer, a row per action.

    Both arrays hold MAX_ACTIONS rows, so both stop there: the mask's slice
    clamps and this bound matches it. Slots past the descriptor list keep the
    zeros the buffer starts with.
    """
    buf = bytearray(MAX_ACTIONS * ACTION_FEATURES)
    msg_type = obs.msg_type
    for i, d in enumerate(obs.action_descriptors[:MAX_ACTIONS]):
        # `_row_fields` is keyed by the packer's own parameter names, so the
        # two cannot fall out of order, and a field it grows that the packer
        # lacks raises rather than packing as a default.
        f = _row_fields(d, i, msg_type, obs)
        pack_action_into(buf, i * ACTION_FEATURES, msg_type, **f)
    return np.frombuffer(buf, dtype=np.uint8).reshape(MAX_ACTIONS, ACTION_FEATURES)


def encode_observation(obs: YuGiOhObservation) -> dict[str, np.ndarray]:
    """Structured observation -> the six arrays the network reads."""
    return {
        "cards": _encode_cards(obs),
        "global_state": _encode_global(obs),
        "actions": _encode_actions(obs),
        "action_mask": _encode_action_mask(obs),
        "pending_chain": obs.pending_chain,
        "event_history": obs.event_history,
    }
