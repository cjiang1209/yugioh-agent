"""Shared observation encoding primitives for RL."""

from __future__ import annotations

import struct

import numpy as np

# ─── Observation dimensions ──────────────────────────────────────────────────
MAX_CARDS = 200
CARD_FEATURES = 42
GLOBAL_FEATURES = 21
MAX_ACTIONS = 32
ACTION_FEATURES = 28
MAX_PENDING_CHAIN = 8
CHAIN_ENTRY_FEATURES = 16
MAX_EVENT_HISTORY = 32
EVENT_ENTRY_FEATURES = 30

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


def encode_u32(val: int) -> tuple[int, int, int, int]:
    """Encode a uint32 value as four uint8 bytes (little-endian)."""
    return val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF


class Layout:
    """A packed row: its fields, and everything derived from them.

    The format string that packs a row, the offsets that read it and the width
    each read takes all come off the one table, so widening a field carries
    its readers with it. Little-endian and unpadded. Iterating a layout yields
    its (name, struct code) pairs, so it reads as the table it declares.
    """

    __slots__ = ("fields", "struct", "offsets", "_readers")

    def __init__(self, *fields: tuple[str, str]) -> None:
        self.fields = fields
        self.struct = struct.Struct("<" + "".join(code for _, code in fields))
        self.offsets: dict[str, int] = {}
        self._readers: dict[str, struct.Struct] = {}
        offset = 0
        for name, code in fields:
            self._readers[name] = reader = struct.Struct("<" + code)
            self.offsets[name] = offset
            offset += reader.size

    def __iter__(self):
        return iter(self.fields)

    def span(self, name: str) -> tuple[int, int]:
        """Where a field starts and how many bytes it spans."""
        return self.offsets[name], self._readers[name].size

    def read(self, row, name: str) -> int:
        """One field, at its declared offset and its declared width.

        Takes anything with a buffer: bytes, a bytearray, or a contiguous
        uint8 row. A batched tensor is not one, so the network's decoders
        derive their own readers from the same table.
        """
        return self._readers[name].unpack_from(row, self.offsets[name])[0]

    def write(self, row, name: str, value: int) -> None:
        """Overwrite one field in place, at its declared offset and width."""
        self._readers[name].pack_into(row, self.offsets[name], value)


# A row may be wider than its fields; the tail is then padding.
#
# Each packer narrows its arguments to fit: a count clamps to its field's
# maximum, an id or bitmask keeps only the low bits. An action's category and
# num_selected are the exceptions, passed through so a value too wide raises.
#
# A layout whose rows are packed as a block takes a `pack_*_into(buf, offset)`
# so the block costs one allocation; the rest return their own row.
CARD_LAYOUT = Layout(
    ("code", "I"),
    ("location", "B"),
    ("sequence", "B"),
    ("position", "B"),
    ("controller", "B"),
    ("is_public", "B"),
    ("card_type", "I"),
    ("level", "B"),
    ("attribute", "B"),
    ("race", "I"),
    ("attack", "H"),
    ("defense", "H"),
    ("lscale", "B"),
    ("rscale", "B"),
    ("link_marker", "H"),
    ("counter_count", "B"),
    ("negated", "B"),
    ("is_overlay", "B"),  # no producer; holds the layout's shape and stays zero
)

# extra_idx_0/1 have no producer; they hold the layout's shape and stay zero.
ACTION_LAYOUT = Layout(
    ("msg_type", "B"),
    ("category", "B"),
    ("code", "I"),  # card passcode
    ("controller", "B"),  # relativized: 0=agent, 1=opponent
    ("location", "B"),
    ("sequence", "H"),
    ("subsequence", "B"),  # the material's slot in an Xyz stack
    ("position", "B"),  # bitmask
    ("direct_attackable", "B"),
    ("param", "B"),
    ("counter_type", "B"),
    ("counter_count", "B"),
    ("index", "B"),
    ("num_selected", "B"),  # cards in the combo so far, 1 for a single pick
    ("extra_idx_0", "B"),
    ("extra_idx_1", "B"),
    ("desc", "Q"),  # engine effect string id
)

# The per-zone card counts, agent's seat first. Named here so the layout and
# the reader that walks them stay one list.
GLOBAL_ZONE_COUNTS = tuple(
    f"{seat}_{zone}"
    for seat in ("my", "opp")
    for zone in ("deck", "hand", "grave", "banished", "extra")
)

# msg_type has no consumer: the decoder skips it, and the per-action-row copy
# is the one the network reads. It holds its byte so the counts that follow
# keep their offsets.
GLOBAL_LAYOUT = Layout(
    ("my_lp", "H"),
    ("opp_lp", "H"),
    ("turn", "B"),
    ("phase", "H"),  # bitmask reaching 0x200, so one byte would drop MAIN2/END
    ("is_my_turn", "B"),
    ("chain_count", "B"),
    ("msg_type", "B"),
    *((name, "B") for name in GLOBAL_ZONE_COUNTS),
    ("is_finished", "B"),
)

CHAIN_LAYOUT = Layout(
    ("code", "I"),
    ("desc", "Q"),
    ("controller", "B"),  # relativized: 0=agent, 1=opponent
    ("location", "B"),  # bitmask
    ("sequence", "B"),
    ("chain_link", "B"),  # 1-based
)

# A tagged record: msg_type is the raw engine MSG id and the discriminator,
# and msg_type == 0 marks an empty slot. controller and turn_player are raw
# engine seats here, relativized by the caller that knows which seat the agent
# holds. hint_type tells apart the hint kinds that share msg_type == MSG_HINT,
# and is 0 for a non-hint entry. Fields a given event kind does not carry stay
# zero.
EVENT_LAYOUT = Layout(
    ("msg_type", "B"),
    ("controller", "B"),
    ("turn_player", "B"),
    ("phase", "B"),  # bit index 0..9, not the raw bitmask
    ("turn_count", "B"),
    ("card_code", "I"),
    ("location", "B"),
    ("sequence", "B"),
    ("target_code", "I"),  # attack
    ("target_location", "B"),  # attack
    ("target_sequence", "B"),  # attack
    ("desc", "Q"),  # chaining
    ("hint_type", "B"),
    ("hint_value", "I"),  # number/race/attribute hint
)


def pack_card_into(
    buf,
    offset: int,
    code: int,
    location: int,
    sequence: int,
    position: int,
    controller: int,
    is_public: bool,
    card_type: int,
    level: int,
    attribute: int,
    race: int,
    attack: int,
    defense: int,
    lscale: int,
    rscale: int,
    link_marker: int,
    counter_count: int,
    negated: bool,
    is_overlay: bool,
) -> None:
    """Write one card's bytes into `buf` at `offset`.

    A negative ATK or DEF -- the `?` sentinel -- reads as 0. A negative
    anywhere else is a caller bug: the clamped fields raise on one, the
    masked fields wrap it into range.
    """
    CARD_LAYOUT.struct.pack_into(
        buf,
        offset,
        code & 0xFFFFFFFF,
        location & 0xFF,
        min(sequence, 255),
        position & 0xFF,
        controller & 0xFF,
        1 if is_public else 0,
        card_type & 0xFFFFFFFF,
        min(level, 255),
        attribute & 0xFF,
        race & 0xFFFFFFFF,
        min(65535, max(0, attack)),
        min(65535, max(0, defense)),
        min(lscale, 255),
        min(rscale, 255),
        link_marker & 0xFFFF,
        min(counter_count, 255),
        1 if negated else 0,
        1 if is_overlay else 0,
    )


def pack_action_into(
    buf,
    offset: int,
    msg_type: int,
    category: int,
    code: int,
    controller: int,
    location: int,
    sequence: int,
    subsequence: int,
    position: int,
    direct_attackable: bool,
    param: int,
    counter_type: int,
    counter_count: int,
    index: int,
    num_selected: int,
    desc: int,
) -> None:
    """Write one action's bytes into `buf` at `offset`.

    Arguments run in layout order, minus the extra_idx slots.

    Two fields lose information: `param` aliases a tribute's release_param
    with a sum prompt's param, and `counter_type` keeps only the low byte of
    a u16 counter id, dropping the COUNTER_* flag bits above 0xFF and id bits
    8-11 that several real counters use. The engine response carries per-card
    counts and no counter id, so only what the network sees is narrowed.
    """
    ACTION_LAYOUT.struct.pack_into(
        buf,
        offset,
        msg_type & 0xFF,
        category,
        code & 0xFFFFFFFF,
        controller & 0xFF,
        location & 0xFF,
        min(sequence, 65535),
        subsequence & 0xFF,
        position & 0xFF,
        1 if direct_attackable else 0,
        param & 0xFF,
        counter_type & 0xFF,
        counter_count & 0xFF,
        index & 0xFF,
        num_selected,
        0,
        0,
        desc & 0xFFFFFFFFFFFFFFFF,
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
    buf = bytearray(CARD_FEATURES)
    pack_card_into(
        buf,
        0,
        code,
        location,
        sequence,
        position,
        controller,
        is_public,
        card_type,
        level,
        attribute,
        race,
        attack,
        defense,
        lscale,
        rscale,
        link_marker,
        counter_count,
        negated,
        is_overlay,
    )
    return np.frombuffer(buf, dtype=np.uint8)


def encode_global(
    my_lp: int,
    opp_lp: int,
    turn: int,
    phase: int,
    is_my_turn: bool,
    chain_count: int,
    msg_type: int,
    my_deck: int,
    my_hand: int,
    my_grave: int,
    my_banished: int,
    my_extra: int,
    opp_deck: int,
    opp_hand: int,
    opp_grave: int,
    opp_banished: int,
    opp_extra: int,
    is_finished: bool,
) -> np.ndarray:
    """Encode the global state as a uint8 feature vector.

    Both producers -- the in-process encoder and the MUD client -- go through
    here, so the two cannot drift apart.
    """
    buf = bytearray(GLOBAL_FEATURES)
    GLOBAL_LAYOUT.struct.pack_into(
        buf,
        0,
        min(my_lp, 65535),
        min(opp_lp, 65535),
        min(turn, 255),
        phase & 0xFFFF,
        1 if is_my_turn else 0,
        min(chain_count, 255),
        msg_type & 0xFF,
        min(my_deck, 255),
        min(my_hand, 255),
        min(my_grave, 255),
        min(my_banished, 255),
        min(my_extra, 255),
        min(opp_deck, 255),
        min(opp_hand, 255),
        min(opp_grave, 255),
        min(opp_banished, 255),
        min(opp_extra, 255),
        1 if is_finished else 0,
    )
    return np.frombuffer(buf, dtype=np.uint8)


def encode_chain_entry(
    code: int,
    desc: int,
    controller: int,
    location: int,
    sequence: int,
    chain_link: int,
) -> np.ndarray:
    """Encode one pending chain entry as a uint8 feature vector."""
    buf = bytearray(CHAIN_ENTRY_FEATURES)
    CHAIN_LAYOUT.struct.pack_into(
        buf,
        0,
        code & 0xFFFFFFFF,
        desc & 0xFFFFFFFFFFFFFFFF,
        controller & 0xFF,
        location & 0xFF,
        min(sequence, 255),
        chain_link & 0xFF,
    )
    return np.frombuffer(buf, dtype=np.uint8)


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

    `EVENT_LAYOUT` carries the meaning of the tag and of each field.
    """
    buf = bytearray(EVENT_ENTRY_FEATURES)
    EVENT_LAYOUT.struct.pack_into(
        buf,
        0,
        msg_type & 0xFF,
        controller & 0xFF,
        turn_player & 0xFF,
        phase & 0xFF,
        min(turn_count, 255),
        card_code & 0xFFFFFFFF,
        location & 0xFF,
        min(sequence, 255),
        target_code & 0xFFFFFFFF,
        target_location & 0xFF,
        min(target_sequence, 255),
        desc & 0xFFFFFFFFFFFFFFFF,
        hint_type & 0xFF,
        hint_value & 0xFFFFFFFF,
    )
    return np.frombuffer(buf, dtype=np.uint8)
