"""Tests for the new ActionMeta lookup tables.

These tests assert structural invariants — properties that, if violated, would
cause silent runtime bugs. Per-key label correctness is intentionally NOT tested:
the dict literal in `constants.py` is its own spec; a typo-test would be redundant.
"""

from yugioh_core.constants import ATTRIBUTE_NAMES, RACE_NAMES, RPS_NAMES


def test_race_names_keys_are_single_bit_masks():
    """RACE_NAMES is indexed by `1 << bit` in _extract_announce_race_actions —
    a multi-bit key (e.g. RACE_ALL) would silently never match."""
    for mask in RACE_NAMES.keys():
        assert mask > 0
        assert mask & (mask - 1) == 0, f"RACE_NAMES key 0x{mask:x} is not a single-bit mask"


def test_attribute_names_covers_all_seven_attributes():
    """ygopro-core defines exactly 7 attributes (EARTH/WATER/FIRE/WIND/LIGHT/DARK/DIVINE).
    A missing entry produces a placeholder label for valid prompts."""
    assert len(ATTRIBUTE_NAMES) == 7


def test_rps_names_is_one_indexed():
    """The engine uses 1/2/3 for rock/paper/scissors, NOT 0/1/2 —
    off-by-one would silently corrupt every RPS prompt."""
    assert set(RPS_NAMES.keys()) == {1, 2, 3}
