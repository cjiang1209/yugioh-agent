"""Tests for the new ActionMeta lookup tables.

These tests assert structural invariants — properties that, if violated, would
cause silent runtime bugs. Per-key label correctness is intentionally NOT tested:
the dict literal in `constants.py` is its own spec; a typo-test would be redundant.
"""

from yugioh_core.constants import (
    ATTRIBUTE_NAMES,
    IGNORED_TYPE_BITS,
    LINK_MARKER_NAMES,
    MONSTER_TYPE_LABELS,
    RACE_NAMES,
    RPS_NAMES,
    SPELL_TRAP_TYPE_LABELS,
    TYPE_EFFECT,
    TYPE_MONSTER,
    TYPE_NORMAL,
    TYPE_SPELL,
    TYPE_TOKEN,
    TYPE_TRAP,
    TYPE_TUNER,
)


def test_race_names_keys_are_single_bit_masks():
    """RACE_NAMES is indexed by `1 << bit` in _extract_announce_race_actions —
    a multi-bit key (e.g. RACE_ALL) would silently never match."""
    for mask in RACE_NAMES:
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


def _masks(table):
    return [mask for mask, _ in table]


def test_type_label_keys_are_single_bit_masks():
    """Labels are matched with `type_val & mask`; a multi-bit key would match
    partially and print a word the card does not have."""
    for table in (MONSTER_TYPE_LABELS, SPELL_TRAP_TYPE_LABELS):
        for mask in _masks(table):
            assert mask > 0
            assert mask & (mask - 1) == 0, f"key 0x{mask:x} is not a single-bit mask"


def test_ignored_type_bits_are_single_bit_masks():
    for mask in IGNORED_TYPE_BITS:
        assert mask > 0
        assert mask & (mask - 1) == 0, f"key 0x{mask:x} is not a single-bit mask"


def test_label_tables_and_ignored_bits_are_disjoint():
    """A bit may be printed in two contexts (TYPE_RITUAL is), but never both
    printed and ignored — that would make the rendering order-dependent."""
    labelled = set(_masks(MONSTER_TYPE_LABELS)) | set(_masks(SPELL_TRAP_TYPE_LABELS))
    assert labelled & IGNORED_TYPE_BITS == set()


def test_structural_bits_are_never_labels():
    """TYPE_MONSTER/SPELL/TRAP select card_type; printing them would duplicate
    the "Spell"/"Trap" word the typeline already starts with."""
    labelled = set(_masks(MONSTER_TYPE_LABELS)) | set(_masks(SPELL_TRAP_TYPE_LABELS))
    assert labelled & {TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP} == set()


def test_ritual_is_labelled_in_both_contexts():
    """Ritual Monsters and Ritual Spells share bit 0x80. Dropping either entry
    silently renders every Ritual Monster, or every Ritual Spell, without the word."""
    from yugioh_core.constants import TYPE_RITUAL

    assert TYPE_RITUAL in _masks(MONSTER_TYPE_LABELS)
    assert TYPE_RITUAL in _masks(SPELL_TRAP_TYPE_LABELS)


def test_monster_body_labels_come_last():
    """Printed typelines end with Normal or Effect ("Dragon / Synchro / Effect").
    Table order is the render order, so body labels must be the final entries."""
    assert _masks(MONSTER_TYPE_LABELS)[-2:] == [TYPE_NORMAL, TYPE_EFFECT]


def test_token_precedes_tuner_in_monster_labels():
    """Swordsoul Token prints "Wyrm / Token / Tuner": the summon-mechanic group
    renders before the ability group."""
    masks = _masks(MONSTER_TYPE_LABELS)
    assert masks.index(TYPE_TOKEN) < masks.index(TYPE_TUNER)


def test_link_marker_names_covers_all_eight_arrows():
    """A Link monster has up to 8 arrows; a missing entry silently drops one
    from the rosette."""
    assert len(LINK_MARKER_NAMES) == 8
    for mask in LINK_MARKER_NAMES:
        assert mask & (mask - 1) == 0, f"key 0x{mask:x} is not a single-bit mask"
