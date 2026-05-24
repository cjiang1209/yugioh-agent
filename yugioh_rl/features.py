"""Decode raw uint8 observations into float tensors for the neural network.

The observation arrays from the environment use packed uint8 bytes with
multi-byte little-endian fields. This module extracts and normalizes them
into properly-typed float32 tensors suitable for neural network input.
"""

from __future__ import annotations

import torch

from yugioh_core.encoding import PER_CARD_DESC_N_VOCAB, SYSSTRING_VOCAB

# ---------------------------------------------------------------------------
# Card features: (B, 200, 42) uint8 → card_ids (B,200) int, card_feats (B,200,F)
# ---------------------------------------------------------------------------

# Location bits (byte 4)
_LOC_BITS = [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x01]  # hand,mzone,szone,grave,banished,extra,deck

# Position bits (byte 6)
_POS_BITS = [0x01, 0x02, 0x04, 0x08]  # FU-Atk, FD-Atk, FU-Def, FD-Def

# Card type bits (bytes 9-12 as uint32)
_TYPE_BITS = [
    0x1,  # monster
    0x2,  # spell
    0x4,  # trap
    0x10,  # normal
    0x20,  # effect
    0x40,  # fusion
    0x80,  # ritual
    0x100,  # trapmonster
    0x200,  # spirit
    0x400,  # union
    0x800,  # gemini
    0x1000,  # tuner
    0x2000,  # synchro
    0x4000,  # token
    0x8000,  # maximum
    0x10000,  # quickplay
    0x20000,  # continuous
    0x40000,  # equip
    0x80000,  # field
    0x100000,  # counter
    0x200000,  # flip
    0x400000,  # toon
    0x800000,  # xyz
    0x1000000,  # pendulum
    0x2000000,  # spsummon
    0x4000000,  # link
]

# Attribute bits (byte 14)
_ATTR_BITS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]  # earth,water,fire,wind,light,dark,divine

# Race bits (bytes 15-18 as uint32)
_RACE_BITS = [
    0x0001,  # warrior
    0x0002,  # spellcaster
    0x0004,  # fairy
    0x0008,  # fiend
    0x0010,  # zombie
    0x0020,  # machine
    0x0040,  # aqua
    0x0080,  # pyro
    0x0100,  # rock
    0x0200,  # winged_beast
    0x0400,  # plant
    0x0800,  # insect
    0x1000,  # thunder
    0x2000,  # dragon
    0x4000,  # beast
    0x8000,  # beast-warrior
    0x10000,  # dinosaur
    0x20000,  # fish
    0x40000,  # sea_serpent
    0x80000,  # reptile
    0x100000,  # psychic
    0x200000,  # divine
    0x400000,  # creator_god
    0x800000,  # wyrm
    0x1000000,  # cyberse
    0x2000000,  # illusion
    0x4000000,  # cyborg
    0x8000000,  # magical_knight
    0x10000000,  # high_dragon
    0x20000000,  # omega_psychic
    0x40000000,  # celestial_warrior
    0x80000000,  # galaxy
]

# Link marker bits (bytes 25-26 as uint16)
_LINK_BITS = [0x01, 0x02, 0x04, 0x08, 0x20, 0x40, 0x80, 0x100]  # 8 arrows

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
        raw: (B, 200, 42) uint8 tensor

    Returns:
        card_ids: (B, 200) long tensor — card passcodes for embedding
        card_feats: (B, 200, CARD_FEAT_DIM) float32 tensor
    """
    raw = raw.long()

    card_ids = _uint32_le(raw, 0)  # (B, 200) — full uint32 card codes

    feats = []

    # location: byte 4 → 7 binary features
    loc = raw[..., 4]
    feats.append(_extract_bits(loc, _LOC_BITS))  # (B,200,7)

    # sequence: byte 5 → normalized
    feats.append((raw[..., 5].float() / 15.0).unsqueeze(-1))  # (B,200,1)

    # position: byte 6 → 4 binary features
    pos = raw[..., 6]
    feats.append(_extract_bits(pos, _POS_BITS))  # (B,200,4)

    # controller: byte 7
    feats.append(raw[..., 7].float().unsqueeze(-1))  # (B,200,1)

    # is_public: byte 8
    feats.append(raw[..., 8].float().unsqueeze(-1))  # (B,200,1)

    # card_type: bytes 9-12 → uint32 → 26 binary features
    ctype = _uint32_le(raw, 9)
    feats.append(_extract_bits(ctype, _TYPE_BITS))  # (B,200,26)

    # level: byte 13
    feats.append((raw[..., 13].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # attribute: byte 14 → 7 binary features
    attr = raw[..., 14]
    feats.append(_extract_bits(attr, _ATTR_BITS))  # (B,200,7)

    # race: bytes 15-18 → uint32 → 32 binary features
    race = _uint32_le(raw, 15)
    feats.append(_extract_bits(race, _RACE_BITS))  # (B,200,32)

    # ATK: bytes 19-20 → uint16
    atk = _uint16_le(raw, 19)
    feats.append((atk.float() / 5000.0).unsqueeze(-1))  # (B,200,1)

    # DEF: bytes 21-22 → uint16
    dfn = _uint16_le(raw, 21)
    feats.append((dfn.float() / 5000.0).unsqueeze(-1))  # (B,200,1)

    # lscale: byte 23
    feats.append((raw[..., 23].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # rscale: byte 24
    feats.append((raw[..., 24].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # link_marker: bytes 25-26 → uint16 → 8 binary features
    lmark = _uint16_le(raw, 25)
    feats.append(_extract_bits(lmark, _LINK_BITS))  # (B,200,8)

    # counter_count: byte 27
    feats.append((raw[..., 27].float() / 10.0).unsqueeze(-1))  # (B,200,1)

    # negated: byte 28
    feats.append(raw[..., 28].float().unsqueeze(-1))  # (B,200,1)

    # is_overlay: byte 29
    feats.append(raw[..., 29].float().unsqueeze(-1))  # (B,200,1)

    card_feats = torch.cat(feats, dim=-1)  # (B, 200, CARD_FEAT_DIM)
    return card_ids, card_feats


# ---------------------------------------------------------------------------
# Global state: (B, 20) uint8 → (B, F_global) float
# ---------------------------------------------------------------------------

# Phase bits (byte 5) — draw=0x01, standby=0x02, main1=0x04, battle_start=0x08,
# battle=0x10, damage=0x20, main2=0x40, end=0x80
_PHASE_BITS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

GLOBAL_FEAT_DIM = 2 + 1 + 8 + 1 + 1 + 10  # = 23
# my_lp(1) + opp_lp(1) + turn(1) + phase(8) + is_my_turn(1) + chain(1) + zone_counts(10)


def decode_global(raw: torch.Tensor) -> torch.Tensor:
    """Decode global state bytes into float features.

    Args:
        raw: (B, 20) uint8 tensor

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

    # phase: byte 5 → 8 binary features
    phase = raw[..., 5]
    feats.append(_extract_bits(phase, _PHASE_BITS))

    # is_my_turn: byte 6
    feats.append(raw[..., 6].float().unsqueeze(-1))

    # chain_count: byte 7
    feats.append((raw[..., 7].float() / 5.0).unsqueeze(-1))

    # zone counts: bytes 9-18 (10 bytes, skip byte 8 = msg_type)
    for i in range(9, 19):
        feats.append((raw[..., i].float() / 40.0).unsqueeze(-1))

    return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Actions: (B, 32, 28) uint8
#   → action_codes (B,32) long, desc_passcodes (B,32) long, desc_ns (B,32) long,
#     action_feats (B,32,F)
# ---------------------------------------------------------------------------

# Heuristic normalizers — module-level constants so they're tunable per-experiment.
# (Rigorous denominators use vocab/dim sizes directly inline.)
_NORM_MSG_TYPE = 255.0  # msg_type byte (heuristic upper bound)
_NORM_CATEGORY = 10.0  # category index (heuristic max)
_NORM_SEQUENCE = 60.0  # sequence (heuristic; deck/banished/GY can grow)
_NORM_SUBSEQUENCE = 15.0  # Xyz overlay slot (heuristic; common stack depth)
_NORM_WEIGHT = 12.0  # tribute release_param / sum.param (≈max monster level)
_NORM_COUNTER_TYPE = 50.0  # counter type id (heuristic; common counters cap)
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
    + 1  # weight
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
        raw: (B, 32, 28) uint8 tensor

    Returns:
        action_codes:   (B, 32) long  — prompt-level card code
        desc_passcodes: (B, 32) long  — high 44 bits of desc (== 0 for sysstring)
        desc_ns:        (B, 32) long  — low 20 bits of desc (per-card slot OR sysstring id)
        action_feats:   (B, 32, ACTION_FEAT_DIM) float32 tensor
    """
    raw = raw.long()

    # code: bytes 2-5 → uint32
    action_codes = _uint32_le(raw, 2)  # (B, 32)

    # desc: bytes 20-27 → uint64; split into passcode (high 44 bits) and n (low 20 bits)
    desc_full = _uint64_le(raw, 20)  # (B, 32) long
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
        (raw[..., 13].float() / _NORM_WEIGHT).unsqueeze(-1),
        (raw[..., 14].float() / _NORM_COUNTER_TYPE).unsqueeze(-1),
        (raw[..., 15].float() / _NORM_COUNTER_COUNT).unsqueeze(-1),
        # index: rigorous (MAX_ACTIONS = 32)
        (raw[..., 16].float() / 32.0).unsqueeze(-1),
        (raw[..., 17].float() / _NORM_NUM_SELECTED).unsqueeze(-1),
        # per_card_desc_n_scalar — masked to 0 on the sysstring path
        per_card_desc_n_scalar.unsqueeze(-1),
    ]

    action_feats = torch.cat(feats, dim=-1)
    # Clamp desc_ns to sysstring vocab range so embedding lookups are safe.
    desc_ns = desc_ns.clamp(max=SYSSTRING_VOCAB - 1)
    return action_codes, desc_passcodes, desc_ns, action_feats
