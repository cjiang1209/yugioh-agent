"""Shared observation encoding primitives for RL."""

from __future__ import annotations

import numpy as np

# ─── Observation dimensions ──────────────────────────────────────────────────
MAX_CARDS = 200
CARD_FEATURES = 42
GLOBAL_FEATURES = 21
MAX_ACTIONS = 32
ACTION_FEATURES = 28
MAX_PENDING_CHAIN = 8
CHAIN_ENTRY_FEATURES = 16

# Vocab sizes for desc_n embeddings (used by yugioh_rl/network.py).
# Per-card desc: rigorous — cards.cdb texts table has str1..str16, so per-card
# desc_n maxes at 15 (16 slots). The model uses a scalar instead of an embedding
# for this branch, so this constant is mainly informational.
PER_CARD_DESC_N_VOCAB = 16

# Sysstring desc: ProjectIgnis ships up to ID ~12125 today, but engine could emit
# values up to u16 max. We use full u16 vocab to future-proof against upstream
# sysstring growth without code change.
SYSSTRING_VOCAB = 65536

# Zone slot allocations per player
ZONE_SLOTS = {
    "hand": 15,
    "mzone": 7,
    "szone": 6,
    "grave": 30,
    "banished": 20,
    "extra": 15,
}
# Total per player = 15+7+6+30+20+15 = 93, times 2 = 186, leaves room for overflow


def encode_u16(val: int) -> tuple[int, int]:
    """Encode a uint16 value as two uint8 bytes (little-endian)."""
    return val & 0xFF, (val >> 8) & 0xFF


def _encode_i16_clamped(val: int) -> tuple[int, int]:
    """Encode a potentially large int as clamped uint16 (0-65535)."""
    val = max(0, min(65535, val))
    return val & 0xFF, (val >> 8) & 0xFF


def encode_u32(val: int) -> tuple[int, int, int, int]:
    """Encode a uint32 value as four uint8 bytes (little-endian)."""
    return val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF


def encode_u64(val: int) -> tuple[int, int, int, int, int, int, int, int]:
    """Encode a uint64 value as eight uint8 bytes (little-endian)."""
    return tuple((val >> (8 * i)) & 0xFF for i in range(8))


def decode_u16(arr, offset: int) -> int:
    """Decode a uint16 LE from two consecutive bytes of arr at *offset*."""
    return int(arr[offset]) | (int(arr[offset + 1]) << 8)


def decode_u32(arr, offset: int) -> int:
    """Decode a uint32 LE from four consecutive bytes of arr at *offset*."""
    return (
        int(arr[offset])
        | (int(arr[offset + 1]) << 8)
        | (int(arr[offset + 2]) << 16)
        | (int(arr[offset + 3]) << 24)
    )


def encode_card(
    code: int,
    location: int,
    sequence: int,
    position: int,
    controller: int,
    is_public: bool,
    card_type: int = 0,
    level: int = 0,
    attribute: int = 0,
    race: int = 0,
    attack: int = 0,
    defense: int = 0,
    lscale: int = 0,
    rscale: int = 0,
    link_marker: int = 0,
    counter_count: int = 0,
    negated: bool = False,
    is_overlay: bool = False,
) -> np.ndarray:
    """Encode a single card as a feature vector.

    Returns:
        np.ndarray of shape (CARD_FEATURES,) dtype uint8
    """
    feat = np.zeros(CARD_FEATURES, dtype=np.uint8)
    idx = 0

    # card_id (4 bytes, uint32 LE)
    feat[idx], feat[idx + 1], feat[idx + 2], feat[idx + 3] = encode_u32(code & 0xFFFFFFFF)
    idx += 4

    # location, sequence, position, controller, is_public
    feat[idx] = location & 0xFF
    idx += 1
    feat[idx] = min(sequence, 255)
    idx += 1
    feat[idx] = position & 0xFF
    idx += 1
    feat[idx] = controller & 0xFF
    idx += 1
    feat[idx] = 1 if is_public else 0
    idx += 1

    # type (4 bytes)
    feat[idx] = card_type & 0xFF
    feat[idx + 1] = (card_type >> 8) & 0xFF
    feat[idx + 2] = (card_type >> 16) & 0xFF
    feat[idx + 3] = (card_type >> 24) & 0xFF
    idx += 4

    # level
    feat[idx] = min(level, 255)
    idx += 1

    # attribute
    feat[idx] = attribute & 0xFF
    idx += 1

    # race (4 bytes, uint32 LE)
    feat[idx], feat[idx + 1], feat[idx + 2], feat[idx + 3] = encode_u32(race & 0xFFFFFFFF)
    idx += 4

    # ATK (2 bytes, clamped)
    feat[idx], feat[idx + 1] = _encode_i16_clamped(attack if attack >= 0 else 0)
    idx += 2

    # DEF (2 bytes, clamped)
    feat[idx], feat[idx + 1] = _encode_i16_clamped(defense if defense >= 0 else 0)
    idx += 2

    # lscale, rscale
    feat[idx] = min(lscale, 255)
    idx += 1
    feat[idx] = min(rscale, 255)
    idx += 1

    # link_marker (2 bytes)
    feat[idx], feat[idx + 1] = encode_u16(link_marker)
    idx += 2

    # counter_count
    feat[idx] = min(counter_count, 255)
    idx += 1

    # negated
    feat[idx] = 1 if negated else 0
    idx += 1

    # is_overlay
    feat[idx] = 1 if is_overlay else 0
    idx += 1

    # Remaining features are padding (zero)
    return feat


def encode_chain_entry(
    code: int,
    desc: int,
    controller: int,
    location: int,
    sequence: int,
    chain_link: int,
) -> np.ndarray:
    """Encode one pending chain entry as a uint8 feature vector.

    Byte layout (16 bytes):
        [0:4]   card_code   (uint32 LE)
        [4:12]  desc        (uint64 LE)
        [12]    controller  (uint8, 0=agent, 1=opponent)
        [13]    location    (uint8 bitmask)
        [14]    sequence    (uint8, clamped to 255)
        [15]    chain_link  (uint8, 1-based)
    """
    feat = np.zeros(CHAIN_ENTRY_FEATURES, dtype=np.uint8)
    feat[0:4] = encode_u32(code & 0xFFFFFFFF)
    feat[4:12] = encode_u64(desc & 0xFFFFFFFFFFFFFFFF)
    feat[12] = controller & 0xFF
    feat[13] = location & 0xFF
    feat[14] = min(sequence, 255)
    feat[15] = chain_link & 0xFF
    return feat


MAX_EVENT_HISTORY = 32
EVENT_ENTRY_FEATURES = 30


def encode_event_entry(
    msg_type: int = 0,
    controller: int = 0,
    turn_player: int = 0,
    phase: int = 0,
    turn_count: int = 0,
    card_code: int = 0,
    location: int = 0,
    sequence: int = 0,
    target_code: int = 0,
    target_location: int = 0,
    target_sequence: int = 0,
    desc: int = 0,
    hint_type: int = 0,
    hint_value: int = 0,
) -> np.ndarray:
    """Encode one event-history entry as a uint8 feature vector.

    Tagged/discriminated record: ``msg_type`` (RAW engine MSG id) is the
    discriminator; ``msg_type == 0`` marks an empty slot. ``controller``/
    ``turn_player`` are RAW engine 0/1 (relativized at encode in
    ``EventHistoryBuffer.to_tensor``). ``hint_type`` disambiguates the hint
    sub-kinds that share ``msg_type == MSG_HINT``.

    Byte layout (role-grouped, fixed offsets):
        [0]      msg_type         discriminator; 0 = empty slot
        [1]      controller       RAW 0/1
        [2]      turn_player      RAW 0/1
        [3]      phase            bit-index 0..9 (not raw bitmask)
        [4]      turn_count       clamped 255
        [5:9]    card_code        uint32 LE
        [9]      location
        [10]     sequence         clamped 255
        [11:15]  target_code      uint32 LE (attack)
        [15]     target_location  (attack)
        [16]     target_sequence  (attack, clamped 255)
        [17:25]  desc             uint64 LE (chaining)
        [25]     hint_type        hint_value discriminator; 0 non-hint
        [26:30]  hint_value       uint32 LE (number/race/attrib hint)
    """
    feat = np.zeros(EVENT_ENTRY_FEATURES, dtype=np.uint8)
    feat[0] = msg_type & 0xFF
    feat[1] = controller & 0xFF
    feat[2] = turn_player & 0xFF
    feat[3] = phase & 0xFF
    feat[4] = min(turn_count, 255)
    feat[5:9] = encode_u32(card_code & 0xFFFFFFFF)
    feat[9] = location & 0xFF
    feat[10] = min(sequence, 255)
    feat[11:15] = encode_u32(target_code & 0xFFFFFFFF)
    feat[15] = target_location & 0xFF
    feat[16] = min(target_sequence, 255)
    feat[17:25] = encode_u64(desc & 0xFFFFFFFFFFFFFFFF)
    feat[25] = hint_type & 0xFF
    feat[26:30] = encode_u32(hint_value & 0xFFFFFFFF)
    return feat
