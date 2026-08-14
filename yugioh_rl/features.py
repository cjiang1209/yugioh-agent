"""Decode raw uint8 observations into float tensors for the neural network.

The observation arrays from the environment use packed uint8 bytes with
multi-byte little-endian fields. This module extracts and normalizes them
into properly-typed float32 tensors suitable for neural network input.
"""

from __future__ import annotations

import torch

from yugioh_core.constants import (
    ATTRIBUTE_DARK,
    ATTRIBUTE_DIVINE,
    ATTRIBUTE_EARTH,
    ATTRIBUTE_FIRE,
    ATTRIBUTE_LIGHT,
    ATTRIBUTE_WATER,
    ATTRIBUTE_WIND,
    HINT_ATTRIB,
    HINT_CODE,
    HINT_NUMBER,
    HINT_RACE,
    LINK_MARKER_BOTTOM,
    LINK_MARKER_BOTTOM_LEFT,
    LINK_MARKER_BOTTOM_RIGHT,
    LINK_MARKER_LEFT,
    LINK_MARKER_RIGHT,
    LINK_MARKER_TOP,
    LINK_MARKER_TOP_LEFT,
    LINK_MARKER_TOP_RIGHT,
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    PHASE_BATTLE,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_DRAW,
    PHASE_END,
    PHASE_MAIN1,
    PHASE_MAIN2,
    PHASE_STANDBY,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    RACE_AQUA,
    RACE_BEAST,
    RACE_BEASTWARRIOR,
    RACE_CELESTIALWARRIOR,
    RACE_CREATORGOD,
    RACE_CYBERSE,
    RACE_CYBORG,
    RACE_DINOSAUR,
    RACE_DIVINE,
    RACE_DRAGON,
    RACE_FAIRY,
    RACE_FIEND,
    RACE_FISH,
    RACE_GALAXY,
    RACE_HIGHDRAGON,
    RACE_ILLUSION,
    RACE_INSECT,
    RACE_MACHINE,
    RACE_MAGICALKNIGHT,
    RACE_OMEGAPSYCHIC,
    RACE_PLANT,
    RACE_PSYCHIC,
    RACE_PYRO,
    RACE_REPTILE,
    RACE_ROCK,
    RACE_SEASERPENT,
    RACE_SPELLCASTER,
    RACE_THUNDER,
    RACE_WARRIOR,
    RACE_WINGEDBEAST,
    RACE_WYRM,
    RACE_ZOMBIE,
    TYPE_CONTINUOUS,
    TYPE_COUNTER,
    TYPE_EFFECT,
    TYPE_EQUIP,
    TYPE_FIELD,
    TYPE_FLIP,
    TYPE_FUSION,
    TYPE_GEMINI,
    TYPE_LINK,
    TYPE_MAXIMUM,
    TYPE_MONSTER,
    TYPE_NORMAL,
    TYPE_PENDULUM,
    TYPE_QUICKPLAY,
    TYPE_RITUAL,
    TYPE_SPELL,
    TYPE_SPIRIT,
    TYPE_SPSUMMON,
    TYPE_SYNCHRO,
    TYPE_TOKEN,
    TYPE_TOON,
    TYPE_TRAP,
    TYPE_TRAPMONSTER,
    TYPE_TUNER,
    TYPE_UNION,
    TYPE_XYZ,
)
from yugioh_core.encoding import (
    MAX_ACTIONS,
    PER_CARD_DESC_N_VOCAB,
    SYSSTRING_VOCAB,
)

# ---------------------------------------------------------------------------
# Card features: (B, MAX_CARDS, CARD_FEATURES) uint8
#   → card_ids (B, MAX_CARDS) int, card_feats (B, MAX_CARDS, F)
# ---------------------------------------------------------------------------

# Location bits (byte 4), in feature-column order
_LOC_BITS = [
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_DECK,
]

# Position bits (byte 6), in feature-column order
_POS_BITS = [
    POS_FACEUP_ATTACK,
    POS_FACEDOWN_ATTACK,
    POS_FACEUP_DEFENSE,
    POS_FACEDOWN_DEFENSE,
]

# Card type bits (bytes 9-12 as uint32)
_TYPE_BITS = [
    TYPE_MONSTER,
    TYPE_SPELL,
    TYPE_TRAP,
    TYPE_NORMAL,
    TYPE_EFFECT,
    TYPE_FUSION,
    TYPE_RITUAL,
    TYPE_TRAPMONSTER,
    TYPE_SPIRIT,
    TYPE_UNION,
    TYPE_GEMINI,
    TYPE_TUNER,
    TYPE_SYNCHRO,
    TYPE_TOKEN,
    TYPE_MAXIMUM,
    TYPE_QUICKPLAY,
    TYPE_CONTINUOUS,
    TYPE_EQUIP,
    TYPE_FIELD,
    TYPE_COUNTER,
    TYPE_FLIP,
    TYPE_TOON,
    TYPE_XYZ,
    TYPE_PENDULUM,
    TYPE_SPSUMMON,
    TYPE_LINK,
]

# Attribute bits (byte 14)
_ATTR_BITS = [
    ATTRIBUTE_EARTH,
    ATTRIBUTE_WATER,
    ATTRIBUTE_FIRE,
    ATTRIBUTE_WIND,
    ATTRIBUTE_LIGHT,
    ATTRIBUTE_DARK,
    ATTRIBUTE_DIVINE,
]

# Race bits (bytes 15-18 as uint32)
_RACE_BITS = [
    RACE_WARRIOR,
    RACE_SPELLCASTER,
    RACE_FAIRY,
    RACE_FIEND,
    RACE_ZOMBIE,
    RACE_MACHINE,
    RACE_AQUA,
    RACE_PYRO,
    RACE_ROCK,
    RACE_WINGEDBEAST,
    RACE_PLANT,
    RACE_INSECT,
    RACE_THUNDER,
    RACE_DRAGON,
    RACE_BEAST,
    RACE_BEASTWARRIOR,
    RACE_DINOSAUR,
    RACE_FISH,
    RACE_SEASERPENT,
    RACE_REPTILE,
    RACE_PSYCHIC,
    RACE_DIVINE,
    RACE_CREATORGOD,
    RACE_WYRM,
    RACE_CYBERSE,
    RACE_ILLUSION,
    RACE_CYBORG,
    RACE_MAGICALKNIGHT,
    RACE_HIGHDRAGON,
    RACE_OMEGAPSYCHIC,
    RACE_CELESTIALWARRIOR,
    RACE_GALAXY,
]

# Link marker bits (bytes 25-26 as uint16), in feature-column order
_LINK_BITS = [
    LINK_MARKER_BOTTOM_LEFT,
    LINK_MARKER_BOTTOM,
    LINK_MARKER_BOTTOM_RIGHT,
    LINK_MARKER_LEFT,
    LINK_MARKER_RIGHT,
    LINK_MARKER_TOP_LEFT,
    LINK_MARKER_TOP,
    LINK_MARKER_TOP_RIGHT,
]

# Number of output float features per card (excluding card_id)
CARD_FEAT_DIM = 7 + 1 + 4 + 1 + 1 + 26 + 1 + 7 + 32 + 1 + 1 + 1 + 1 + 8 + 1 + 1 + 1  # = 95
# location(7) + sequence(1) + position(4) + controller(1) + is_public(1)
# + card_type(26) + level(1) + attribute(7) + race(32) + atk(1) + def(1) + lscale(1) + rscale(1)
# + link_marker(8) + counter(1) + negated(1) + is_overlay(1) = 95


def _uint16_le(raw: torch.Tensor, byte0: int) -> torch.Tensor:
    """Extract uint16 LE from two consecutive uint8 bytes along last dim.

    raw: (..., N) uint8/int tensor.  Returns (...) int tensor.
    """
    return raw[..., byte0].long() + raw[..., byte0 + 1].long() * 256


def _uint32_le(raw: torch.Tensor, byte0: int) -> torch.Tensor:
    """Extract uint32 LE from four consecutive uint8 bytes along last dim."""
    return (
        raw[..., byte0].long()
        + raw[..., byte0 + 1].long() * 256
        + raw[..., byte0 + 2].long() * 65536
        + raw[..., byte0 + 3].long() * 16777216
    )


_SHIFTS_U64 = torch.tensor([1 << (8 * i) for i in range(8)], dtype=torch.int64)


def _uint64_le(raw: torch.Tensor, byte0: int) -> torch.Tensor:
    """Extract uint64 LE from eight consecutive uint8 bytes along last dim.

    raw: (..., N) uint8/int tensor.  Returns (...) int64 tensor.

    Overflow note: signed int64 wraps at 2^63. The action vector's only
    u64 field is `desc = (passcode << 20) | n_low20`. Passcode is u32 so
    desc occupies at most bits 0-51; bytes 26-27 (bits 48+) are always 0
    in practice, keeping the high bit clear and preventing wraparound.
    """
    bytes_slice = raw[..., byte0 : byte0 + 8].long()
    return (bytes_slice * _SHIFTS_U64.to(raw.device)).sum(dim=-1)


def _extract_bits(val: torch.Tensor, bits: list[int]) -> torch.Tensor:
    """Extract binary features from a bitmask.

    val: (...) int tensor.  bits: list of bit masks.
    Returns (..., len(bits)) float tensor.
    """
    parts = []
    for b in bits:
        parts.append(((val & b) != 0).float())
    return torch.stack(parts, dim=-1)


def decode_cards(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode card observation bytes into card IDs and float features.

    Args:
        raw: (B, MAX_CARDS, CARD_FEATURES) uint8 tensor

    Returns:
        card_ids: (B, MAX_CARDS) long tensor — card passcodes for embedding
        card_feats: (B, MAX_CARDS, CARD_FEAT_DIM) float32 tensor
    """
    raw = raw.long()

    card_ids = _uint32_le(raw, 0)  # (B, MAX_CARDS) — full uint32 card codes

    feats = []

    # location: byte 4 → 7 binary features
    loc = raw[..., 4]
    feats.append(_extract_bits(loc, _LOC_BITS))  # (B, MAX_CARDS, 7)

    # sequence: byte 5 → normalized
    feats.append((raw[..., 5].float() / 15.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # position: byte 6 → 4 binary features
    pos = raw[..., 6]
    feats.append(_extract_bits(pos, _POS_BITS))  # (B, MAX_CARDS, 4)

    # controller: byte 7
    feats.append(raw[..., 7].float().unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # is_public: byte 8
    feats.append(raw[..., 8].float().unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # card_type: bytes 9-12 → uint32 → 26 binary features
    ctype = _uint32_le(raw, 9)
    feats.append(_extract_bits(ctype, _TYPE_BITS))  # (B, MAX_CARDS, 26)

    # level: byte 13
    feats.append((raw[..., 13].float() / 12.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # attribute: byte 14 → 7 binary features
    attr = raw[..., 14]
    feats.append(_extract_bits(attr, _ATTR_BITS))  # (B, MAX_CARDS, 7)

    # race: bytes 15-18 → uint32 → 32 binary features
    race = _uint32_le(raw, 15)
    feats.append(_extract_bits(race, _RACE_BITS))  # (B, MAX_CARDS, 32)

    # ATK: bytes 19-20 → uint16
    atk = _uint16_le(raw, 19)
    feats.append((atk.float() / 5000.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # DEF: bytes 21-22 → uint16
    dfn = _uint16_le(raw, 21)
    feats.append((dfn.float() / 5000.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # lscale: byte 23
    feats.append((raw[..., 23].float() / 12.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # rscale: byte 24
    feats.append((raw[..., 24].float() / 12.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # link_marker: bytes 25-26 → uint16 → 8 binary features
    lmark = _uint16_le(raw, 25)
    feats.append(_extract_bits(lmark, _LINK_BITS))  # (B, MAX_CARDS, 8)

    # counter_count: byte 27
    feats.append((raw[..., 27].float() / 10.0).unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # negated: byte 28
    feats.append(raw[..., 28].float().unsqueeze(-1))  # (B, MAX_CARDS, 1)

    # is_overlay: byte 29
    feats.append(raw[..., 29].float().unsqueeze(-1))  # (B, MAX_CARDS, 1)

    card_feats = torch.cat(feats, dim=-1)  # (B, MAX_CARDS, CARD_FEAT_DIM)
    return card_ids, card_feats


# ---------------------------------------------------------------------------
# Global state: (B, GLOBAL_FEATURES) uint8 → (B, F_global) float
# ---------------------------------------------------------------------------

# Phase bits (bytes 5-6, uint16 LE), one column per phase
_PHASE_BITS = [
    PHASE_DRAW,
    PHASE_STANDBY,
    PHASE_MAIN1,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_BATTLE,
    PHASE_MAIN2,
    PHASE_END,
]

GLOBAL_FEAT_DIM = 2 + 1 + 10 + 1 + 1 + 10  # = 25
# my_lp(1) + opp_lp(1) + turn(1) + phase(10) + is_my_turn(1) + chain(1) + zone_counts(10)


def decode_global(raw: torch.Tensor) -> torch.Tensor:
    """Decode global state bytes into float features.

    Args:
        raw: (B, GLOBAL_FEATURES) uint8 tensor

    Returns:
        global_feats: (B, GLOBAL_FEAT_DIM) float32 tensor
    """
    raw = raw.long()
    feats = []

    # my_lp: bytes 0-1
    my_lp = _uint16_le(raw, 0)
    feats.append((my_lp.float() / 8000.0).unsqueeze(-1))

    # opp_lp: bytes 2-3
    opp_lp = _uint16_le(raw, 2)
    feats.append((opp_lp.float() / 8000.0).unsqueeze(-1))

    # turn_count: byte 4
    feats.append((raw[..., 4].float() / 50.0).unsqueeze(-1))

    # phase: bytes 5-6 (uint16 LE) → 10 binary features
    phase = _uint16_le(raw, 5)
    feats.append(_extract_bits(phase, _PHASE_BITS))

    # is_my_turn: byte 7
    feats.append(raw[..., 7].float().unsqueeze(-1))

    # chain_count: byte 8
    feats.append((raw[..., 8].float() / 5.0).unsqueeze(-1))

    # zone counts: bytes 10-19 (10 bytes, skip byte 9 = msg_type)
    for i in range(10, 20):
        feats.append((raw[..., i].float() / 40.0).unsqueeze(-1))

    # Byte 20 (is_finished) is deliberately not decoded, so this stops one byte
    # short of GLOBAL_FEATURES. The rollout loop substitutes a fresh-episode
    # observation at every done index before the next forward, so the network
    # only ever sees is_finished == 0 -- there is no signal in it. Callers that
    # need the terminal flag read ``obs.done``.
    return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Actions: (B, MAX_ACTIONS, ACTION_FEATURES) uint8
#   → action_codes (B, MAX_ACTIONS) long, desc_passcodes (B, MAX_ACTIONS) long, desc_ns (B, MAX_ACTIONS) long,
#     action_feats (B, MAX_ACTIONS, F)
# ---------------------------------------------------------------------------

# Heuristic normalizers — module-level constants so they're tunable per-experiment.
# (Rigorous denominators use vocab/dim sizes directly inline.)
_NORM_MSG_TYPE = 255.0  # msg_type byte (heuristic upper bound)
_NORM_CATEGORY = 10.0  # category index (heuristic max)
_NORM_SEQUENCE = 60.0  # sequence (heuristic; deck/banished/GY can grow)
_NORM_SUBSEQUENCE = 15.0  # Xyz overlay slot (heuristic; common stack depth)
_NORM_PARAM = 12.0  # tribute release_param / sum.param (≈max monster level)
_NORM_COUNTER_TYPE = 255.0  # counter-type byte; its full range, so this cannot exceed 1
_NORM_COUNTER_COUNT = 15.0  # counters on one card (heuristic)
_NORM_NUM_SELECTED = 5.0  # accumulated picks in a multi-step prompt (heuristic)

# Per-card desc_n vocab — rigorous: cards.cdb texts has str1..str16
# (Imported at top of module; re-noted here for context.)

ACTION_FEAT_DIM = (
    1  # msg_type
    + 1  # category
    + 1  # controller (binary)
    + 7  # location bits
    + 1  # sequence
    + 1  # subsequence
    + 4  # position bits
    + 1  # direct_attackable (binary)
    + 1  # param
    + 1  # counter_type
    + 1  # counter_count
    + 1  # index
    + 1  # num_selected
    + 1  # per_card_desc_n_scalar (masked to 0 when sysstring)
)  # = 23


def decode_actions(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode action feature bytes into action codes, desc components, and float feats.

    Args:
        raw: (B, MAX_ACTIONS, ACTION_FEATURES) uint8 tensor

    Returns:
        action_codes:   (B, MAX_ACTIONS) long  — prompt-level card code
        desc_passcodes: (B, MAX_ACTIONS) long  — high 44 bits of desc (== 0 for sysstring)
        desc_ns:        (B, MAX_ACTIONS) long  — low 20 bits of desc (per-card slot OR sysstring id)
        action_feats:   (B, MAX_ACTIONS, ACTION_FEAT_DIM) float32 tensor
    """
    raw = raw.long()

    # code: bytes 2-5 → uint32
    action_codes = _uint32_le(raw, 2)  # (B, MAX_ACTIONS)

    # desc: bytes 20-27 → uint64; split into passcode (high 44 bits) and n (low 20 bits)
    desc_full = _uint64_le(raw, 20)  # (B, MAX_ACTIONS) long
    desc_passcodes = desc_full >> 20
    desc_ns = desc_full & 0xFFFFF

    # Per-card desc_n scalar — meaningful only when desc.passcode != 0.
    # Masked to 0 when sysstring so the MLP gets a clean disambiguated signal.
    is_sysstring_mask = (desc_passcodes == 0).float()
    per_card_desc_n_scalar = (desc_ns.float() / float(PER_CARD_DESC_N_VOCAB - 1)) * (
        1.0 - is_sysstring_mask
    )

    feats = [
        (raw[..., 0].float() / _NORM_MSG_TYPE).unsqueeze(-1),
        (raw[..., 1].float() / _NORM_CATEGORY).unsqueeze(-1),
        # controller: binary 0/1, no normalization
        raw[..., 6].float().unsqueeze(-1),
        # location: 7 rigorous bits
        _extract_bits(raw[..., 7], _LOC_BITS),
        (_uint16_le(raw, 8).float() / _NORM_SEQUENCE).unsqueeze(-1),
        (raw[..., 10].float() / _NORM_SUBSEQUENCE).unsqueeze(-1),
        # position: 4 rigorous bits
        _extract_bits(raw[..., 11], _POS_BITS),
        # direct_attackable: binary 0/1
        raw[..., 12].float().unsqueeze(-1),
        (raw[..., 13].float() / _NORM_PARAM).unsqueeze(-1),
        (raw[..., 14].float() / _NORM_COUNTER_TYPE).unsqueeze(-1),
        (raw[..., 15].float() / _NORM_COUNTER_COUNT).unsqueeze(-1),
        # index: rigorous -- the action slot count
        (raw[..., 16].float() / float(MAX_ACTIONS)).unsqueeze(-1),
        (raw[..., 17].float() / _NORM_NUM_SELECTED).unsqueeze(-1),
        # per_card_desc_n_scalar — masked to 0 on the sysstring path
        per_card_desc_n_scalar.unsqueeze(-1),
    ]

    action_feats = torch.cat(feats, dim=-1)
    # Clamp desc_ns to sysstring vocab range so embedding lookups are safe.
    desc_ns = desc_ns.clamp(max=SYSSTRING_VOCAB - 1)
    return action_codes, desc_passcodes, desc_ns, action_feats


# ── Pending chain decoding ──────────────────────────────────────────────

CHAIN_FEAT_DIM = (
    11  # controller(1) + location(7) + sequence(1) + chain_link(1) + per_card_desc_n(1)
)


def decode_pending_chain(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw pending chain bytes into embedable fields + features.

    Args:
        raw: (B, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES) uint8 tensor

    Returns:
        chain_codes:     (B, MAX_PENDING_CHAIN) long  — card passcodes
        desc_passcodes:  (B, MAX_PENDING_CHAIN) long  — high 44 bits of desc
        desc_ns:         (B, MAX_PENDING_CHAIN) long  — low 20 bits of desc
        chain_feats:     (B, MAX_PENDING_CHAIN, CHAIN_FEAT_DIM) float32
    """
    raw_long = raw.long()

    chain_codes = _uint32_le(raw_long, 0)  # bytes 0-3

    desc_full = _uint64_le(raw_long, 4)  # bytes 4-11
    desc_passcodes = desc_full >> 20
    desc_ns = desc_full & 0xFFFFF

    feats = []

    # controller (byte 12): scalar float
    feats.append(raw_long[:, :, 12].float().unsqueeze(-1))

    # location (byte 13): 7-bit one-hot using _extract_bits
    loc_byte = raw_long[:, :, 13]
    feats.append(_extract_bits(loc_byte, _LOC_BITS))  # (B, MAX_PENDING_CHAIN, 7)

    # sequence (byte 14): scalar float
    feats.append(raw_long[:, :, 14].float().unsqueeze(-1))

    # chain_link (byte 15): scalar float
    feats.append(raw_long[:, :, 15].float().unsqueeze(-1))

    # per_card_desc_n: derived from desc_ns, masked to 0 when sysstring
    # (same pattern as decode_actions)
    is_sysstring_mask = (desc_passcodes == 0).float()
    per_card_desc_n_scalar = (desc_ns.float() / float(PER_CARD_DESC_N_VOCAB - 1)) * (
        1.0 - is_sysstring_mask
    )
    feats.append(per_card_desc_n_scalar.unsqueeze(-1))

    chain_feats = torch.cat(feats, dim=-1)  # (B, MAX_PENDING_CHAIN, CHAIN_FEAT_DIM)
    return chain_codes, desc_passcodes, desc_ns, chain_feats


# ── Event history decoding ──────────────────────────────────────────────

# Declaration-hint ids → 4 one-hot columns.
_EVENT_HINT_IDS = [HINT_RACE, HINT_ATTRIB, HINT_CODE, HINT_NUMBER]

# Float features per event entry. msg_type and phase are embedded via aux_ids;
# everything else is fed here directly, in this column order (matches the
# torch.cat in decode_event_history):
#   scalar_feats(6): controller, turn_player, sequence, target_sequence,
#                    hint_value, turn_delta
#   + location(7 one-hot) + target_location(7 one-hot) + hint_type(4 one-hot)
EVENT_FEAT_DIM = 6 + 2 * len(_LOC_BITS) + len(_EVENT_HINT_IDS)  # = 24


def decode_event_history(raw: torch.Tensor):
    """Decode (B, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES) uint8 event-history bytes (tagged/discriminated record).

    Returns (codes, desc_passcodes, desc_ns, target_codes, aux_ids, feats).
    ``aux_ids`` carries only the embedded categoricals ``[msg_type, phase]``.
    The remaining categoricals are encoded directly into ``feats``: location /
    target_location as one-hot over ``_LOC_BITS`` (a card is in exactly one
    zone; bit-expanded rather than fed as the raw bitmask scalar); hint_type as
    one-hot over the four declaration ids; controller / turn_player as raw 0/1
    scalars (already relativized to agent=0/opp=1 at encode time).
    turn_delta is computed relative to the newest event's turn_count in each row
    (== current turn at encode), clamped to [0,16].
    Byte offsets match the entry encoder: msg_type[0], controller[1],
    turn_player[2], phase[3], turn_count[4], card_code[5:9], location[9],
    sequence[10], target_code[11:15], target_location[15], target_sequence[16],
    desc[17:25], hint_type[25], hint_value[26:30].
    """
    raw_long = raw.long()
    msg_type = raw_long[..., 0]
    controller = raw_long[..., 1].float()
    turn_player = raw_long[..., 2].float()
    phase = raw_long[..., 3]
    turn_count = raw_long[..., 4]
    codes = _uint32_le(raw_long, 5)
    location = _extract_bits(raw_long[..., 9], _LOC_BITS)  # (B,T,7) one-hot zone
    sequence = raw_long[..., 10].float()
    target_codes = _uint32_le(raw_long, 11)
    target_location = _extract_bits(raw_long[..., 15], _LOC_BITS)  # (B,T,7) one-hot zone
    target_sequence = raw_long[..., 16].float()
    desc_full = _uint64_le(raw_long, 17)
    desc_passcodes = desc_full >> 20
    desc_ns = desc_full & 0xFFFFF
    hint_type = raw_long[..., 25]
    hint_value = _uint32_le(raw_long, 26).float()

    # hint_type is nominal (race/attrib/code/number) → one-hot; empty/non-hint
    # entries (hint_type not in the set) get an all-zero row.
    hint_onehot = torch.stack(
        [(hint_type == hid).float() for hid in _EVENT_HINT_IDS], dim=-1
    )  # (B,T,4)

    nonempty = msg_type != 0
    # newest turn = max turn_count over entries; empty rows are all-zero bytes
    # so their turn_count is already 0 and never wins the max.
    # NOTE: at a turn boundary where the current turn has recorded no event yet,
    # this reference lags the true turn, so deltas are uniformly too small (a
    # global shift; relative recency is preserved, self-corrects on next event).
    cur_turn = turn_count.amax(dim=1, keepdim=True)
    turn_delta = (cur_turn - turn_count).clamp(min=0, max=16).float()
    turn_delta = torch.where(nonempty, turn_delta, torch.zeros_like(turn_delta))

    aux_ids = torch.stack([msg_type, phase], dim=-1)
    scalar_feats = torch.stack(
        [controller, turn_player, sequence, target_sequence, hint_value, turn_delta],
        dim=-1,
    )  # (B,T,6)
    feats = torch.cat([scalar_feats, location, target_location, hint_onehot], dim=-1)
    # Clamp desc_ns to sysstring vocab range so embedding lookups are safe.
    desc_ns = desc_ns.clamp(max=SYSSTRING_VOCAB - 1)
    return codes, desc_passcodes, desc_ns, target_codes, aux_ids, feats
