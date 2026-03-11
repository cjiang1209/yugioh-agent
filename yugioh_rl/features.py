"""Decode raw uint8 observations into float tensors for the neural network.

The observation arrays from the environment use packed uint8 bytes with
multi-byte little-endian fields. This module extracts and normalizes them
into properly-typed float32 tensors suitable for neural network input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Card features: (B, 200, 42) uint8 → card_ids (B,200) int, card_feats (B,200,F)
# ---------------------------------------------------------------------------

# Location bits (byte 2)
_LOC_BITS = [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x01]  # hand,mzone,szone,grave,banished,extra,deck

# Position bits (byte 4)
_POS_BITS = [0x01, 0x02, 0x04, 0x08]  # FU-Atk, FD-Atk, FU-Def, FD-Def

# Card type bits (bytes 7-10 as uint32)
_TYPE_BITS = [
    0x1,        # monster
    0x2,        # spell
    0x4,        # trap
    0x40,       # fusion
    0x2000,     # synchro
    0x800000,   # xyz
    0x4000000,  # link
    0x20,       # effect
]

# Attribute bits (byte 12)
_ATTR_BITS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]  # earth,water,fire,wind,light,dark,divine

# Race bits (bytes 13-14 as uint16)
_RACE_BITS = [
    0x0001, 0x0002, 0x0004, 0x0008,  # warrior, spellcaster, fairy, fiend
    0x0010, 0x0020, 0x0040, 0x0080,  # zombie, machine, aqua, pyro
    0x0100, 0x0200, 0x0400, 0x0800,  # rock, winged_beast, plant, insect
    0x1000, 0x2000, 0x4000, 0x8000,  # thunder, dragon, beast, beast-warrior
]

# Link marker bits (bytes 21-22 as uint16)
_LINK_BITS = [0x01, 0x02, 0x04, 0x08, 0x20, 0x40, 0x80, 0x100]  # 8 arrows

# Number of output float features per card (excluding card_id)
CARD_FEAT_DIM = 7 + 1 + 4 + 1 + 1 + 8 + 1 + 7 + 16 + 1 + 1 + 1 + 1 + 8 + 1 + 1 + 1  # = 61
# location(7) + sequence(1) + position(4) + controller(1) + is_public(1)
# + card_type(8) + level(1) + attribute(7) + race(16) + atk(1) + def(1) + lscale(1) + rscale(1)
# + link_marker(8) + counter(1) + negated(1) + is_overlay(1) = 61


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
    B = raw.shape[0]
    raw = raw.long()

    card_ids = _uint16_le(raw, 0)  # (B, 200)

    feats = []

    # location: byte 2 → 7 binary features
    loc = raw[..., 2]
    feats.append(_extract_bits(loc, _LOC_BITS))  # (B,200,7)

    # sequence: byte 3 → normalized
    feats.append((raw[..., 3].float() / 15.0).unsqueeze(-1))  # (B,200,1)

    # position: byte 4 → 4 binary features
    pos = raw[..., 4]
    feats.append(_extract_bits(pos, _POS_BITS))  # (B,200,4)

    # controller: byte 5
    feats.append(raw[..., 5].float().unsqueeze(-1))  # (B,200,1)

    # is_public: byte 6
    feats.append(raw[..., 6].float().unsqueeze(-1))  # (B,200,1)

    # card_type: bytes 7-10 → uint32 → 8 binary features
    ctype = _uint32_le(raw, 7)
    feats.append(_extract_bits(ctype, _TYPE_BITS))  # (B,200,8)

    # level: byte 11
    feats.append((raw[..., 11].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # attribute: byte 12 → 7 binary features
    attr = raw[..., 12]
    feats.append(_extract_bits(attr, _ATTR_BITS))  # (B,200,7)

    # race: bytes 13-14 → uint16 → 16 binary features
    race = _uint16_le(raw, 13)
    feats.append(_extract_bits(race, _RACE_BITS))  # (B,200,16)

    # ATK: bytes 15-16 → uint16
    atk = _uint16_le(raw, 15)
    feats.append((atk.float() / 5000.0).unsqueeze(-1))  # (B,200,1)

    # DEF: bytes 17-18 → uint16
    dfn = _uint16_le(raw, 17)
    feats.append((dfn.float() / 5000.0).unsqueeze(-1))  # (B,200,1)

    # lscale: byte 19
    feats.append((raw[..., 19].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # rscale: byte 20
    feats.append((raw[..., 20].float() / 12.0).unsqueeze(-1))  # (B,200,1)

    # link_marker: bytes 21-22 → uint16 → 8 binary features
    lmark = _uint16_le(raw, 21)
    feats.append(_extract_bits(lmark, _LINK_BITS))  # (B,200,8)

    # counter_count: byte 23
    feats.append((raw[..., 23].float() / 10.0).unsqueeze(-1))  # (B,200,1)

    # negated: byte 24
    feats.append(raw[..., 24].float().unsqueeze(-1))  # (B,200,1)

    # is_overlay: byte 25
    feats.append(raw[..., 25].float().unsqueeze(-1))  # (B,200,1)

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
# Actions: (B, 32, 12) uint8 → action_codes (B,32) int, action_feats (B,32,F)
# ---------------------------------------------------------------------------

ACTION_FEAT_DIM = 1 + 1 + 7 + 1 + 1 + 1  # = 12
# msg_type(1) + category(1) + location(7) + sequence(1) + index(1) + num_selected(1)


def decode_actions(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode action feature bytes into action codes and float features.

    Args:
        raw: (B, 32, 12) uint8 tensor

    Returns:
        action_codes: (B, 32) long tensor — card codes for embedding
        action_feats: (B, 32, ACTION_FEAT_DIM) float32 tensor
    """
    raw = raw.long()

    # code: bytes 2-5 → uint32
    action_codes = _uint32_le(raw, 2)  # (B, 32)

    feats = []

    # msg_type: byte 0
    feats.append((raw[..., 0].float() / 255.0).unsqueeze(-1))

    # category: byte 1
    feats.append((raw[..., 1].float() / 10.0).unsqueeze(-1))

    # location: byte 6 → 7 binary features
    loc = raw[..., 6]
    feats.append(_extract_bits(loc, _LOC_BITS))

    # sequence: byte 7
    feats.append((raw[..., 7].float() / 15.0).unsqueeze(-1))

    # index: byte 8
    feats.append((raw[..., 8].float() / 32.0).unsqueeze(-1))

    # num_selected: byte 9
    feats.append((raw[..., 9].float() / 5.0).unsqueeze(-1))

    action_feats = torch.cat(feats, dim=-1)
    return action_codes, action_feats
