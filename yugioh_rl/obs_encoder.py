"""Pack a structured observation into the uint8 arrays the network reads.

Feature encoding is network-specific, so it lives here rather than on the
observation, which stays a data carrier. Called by ``TrainingEnv`` for
collection and ``NetworkOpponent`` for inference.

Every clamp and mask in this module exists to fit a uint8 array. The structured
models hold raw engine values; narrowing happens here and nowhere else.
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
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    encode_card,
    encode_u16,
    encode_u32,
    encode_u64,
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
    cards = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
    for i, c in enumerate(obs.card_states[:MAX_CARDS]):
        cards[i] = encode_card(
            code=c.code,
            location=c.location,
            sequence=c.sequence,
            position=c.position,
            controller=c.controller,
            is_public=c.is_public,
            card_type=c.card_type,
            level=c.level,
            attribute=c.attribute,
            race=c.race & 0xFFFFFFFF,
            attack=c.attack,
            defense=c.defense,
            lscale=c.lscale,
            rscale=c.rscale,
            link_marker=c.link_marker,
            counter_count=c.counter_count,
            negated=c.negated,
            is_overlay=c.is_overlay,
        )
    return cards


def _encode_global(obs: YuGiOhObservation) -> np.ndarray:
    g = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
    s = obs.global_
    idx = 0
    g[idx], g[idx + 1] = encode_u16(min(s.my_lp, 65535))
    idx += 2
    g[idx], g[idx + 1] = encode_u16(min(s.opp_lp, 65535))
    idx += 2
    g[idx] = min(s.turn, 255)
    idx += 1
    g[idx], g[idx + 1] = encode_u16(s.phase)
    idx += 2
    g[idx] = 1 if s.is_my_turn else 0
    idx += 1
    g[idx] = min(s.chain_count, 255)
    idx += 1
    g[idx] = s.msg_type & 0xFF
    idx += 1
    for count in (
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
    ):
        g[idx] = min(count, 255)
        idx += 1
    g[idx] = 1 if s.is_finished else 0
    return g


def _num_actions(obs: YuGiOhObservation) -> int:
    """Number of legal actions.

    ``action_descriptors`` is padded with ``None`` up to ``MAX_ACTIONS``.
    Count the real entries rather than the list length, or every slot reads
    as legal. The padding is a contiguous suffix, so this count is also the
    prefix length.
    """
    return sum(1 for d in obs.action_descriptors[:MAX_ACTIONS] if d is not None)


def _encode_action_mask(obs: YuGiOhObservation) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[: _num_actions(obs)] = 1
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
    """Reconstruct the raw values `_encode_action`'s byte layout needs, one
    branch per descriptor kind. Defaults mirror `action.get(key, 0)` --
    num_selected defaults to 1, everything else to 0.
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


def _encode_action_row(d, i: int, msg_type: int, obs: YuGiOhObservation) -> np.ndarray:
    f = _row_fields(d, i, msg_type, obs)
    feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
    feat[0] = msg_type & 0xFF
    feat[1] = f["category"]
    feat[2:6] = encode_u32(f["code"] & 0xFFFFFFFF)
    feat[6] = f["controller"] & 0xFF
    feat[7] = f["location"] & 0xFF
    feat[8:10] = encode_u16(min(f["sequence"], 65535))
    feat[10] = f["subsequence"] & 0xFF
    feat[11] = f["position"] & 0xFF
    feat[12] = 1 if f["direct_attackable"] else 0
    feat[13] = f["param"] & 0xFF
    feat[14] = f["counter_type"] & 0xFF
    feat[15] = f["counter_count"] & 0xFF
    feat[16] = f["index"] & 0xFF
    feat[17] = f["num_selected"]
    # feat[18:20] (extra_idx) have no producer -- left zero.
    feat[20:28] = encode_u64(f["desc"] & 0xFFFFFFFFFFFFFFFF)
    return feat


def _encode_actions(obs: YuGiOhObservation) -> np.ndarray:
    feats = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
    msg_type = obs.msg_type
    for i, d in enumerate(obs.action_descriptors[:MAX_ACTIONS]):
        if d is None:
            continue
        feats[i] = _encode_action_row(d, i, msg_type, obs)
    return feats


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
