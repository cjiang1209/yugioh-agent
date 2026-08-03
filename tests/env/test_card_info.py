"""Decode tests for build_card_info.

Hermetic — a duck-typed fake card_db supplies rows, so these run without
cards.cdb. Every row below is real data, read from assets/cards.cdb.

The bit-partition guard in this file is the counterpart: it queries the live
database to prove the label tables the fakes exercise still cover every card.
"""

from yugioh_env.server.card_info import build_card_info


def _row(**over):
    """A get_card() row with monster defaults; override per card."""
    row = {
        "code": 0,
        "alias": 0,
        "setcodes": [],
        "type": 0x21,  # MONSTER | EFFECT
        "level": 4,
        "attribute": 0x20,  # DARK
        "race": 0x2000,  # Dragon
        "attack": 1000,
        "defense": 1000,
        "lscale": 0,
        "rscale": 0,
        "link_marker": 0,
    }
    row.update(over)
    return row


class _FakeCardDB:
    """Only the three accessors build_card_info calls."""

    def __init__(self, row, name="Test Card", desc="text"):
        self._row = row
        self._name = name
        self._desc = desc

    def get_card(self, code):
        return self._row

    def get_card_name(self, code):
        return self._name

    def get_card_desc(self, code):
        return self._desc


def test_normal_monster():
    """Blue-Eyes White Dragon (89631139, 0x11) — LIGHT Dragon, level 8."""
    info = build_card_info(
        89631139,
        _FakeCardDB(
            _row(type=0x11, level=8, attribute=0x10, race=0x2000, attack=3000, defense=2500),
            name="Blue-Eyes White Dragon",
        ),
    )
    assert info.card_type == "monster"
    assert info.typeline == ["Dragon", "Normal"]
    assert info.attribute == "LIGHT"
    assert info.race == "Dragon"
    assert info.level == 8
    assert info.level_kind == "level"
    assert info.attack == 3000
    assert info.defense == 2500
    assert info.scales is None
    assert info.link_arrows is None


def test_synchro_monster():
    """Stardust Dragon (44508094, 0x2021) — mechanic label before body label."""
    info = build_card_info(
        44508094,
        _FakeCardDB(
            _row(type=0x2021, level=8, attribute=0x08, race=0x2000, attack=2500, defense=2000),
            name="Stardust Dragon",
        ),
    )
    assert info.typeline == ["Dragon", "Synchro", "Effect"]
    assert info.attribute == "WIND"


def test_synchro_tuner_monster():
    """Zalen the Shackled Dragon (4891376, 0x3021) — Tuner sits between the
    mechanic and body labels: "Dragon / Synchro / Tuner / Effect"."""
    info = build_card_info(
        4891376,
        _FakeCardDB(
            _row(type=0x3021, level=7, attribute=0x20, race=0x2000, attack=2800, defense=2100),
            name="Zalen the Shackled Dragon",
        ),
    )
    assert info.typeline == ["Dragon", "Synchro", "Tuner", "Effect"]


def test_pendulum_monster_has_scales():
    """D/D Savant Thomas (41546, 0x1000021), packed level 0x6060008 → scales 6/6."""
    info = build_card_info(
        41546,
        _FakeCardDB(
            _row(
                type=0x1000021,
                level=8,
                attribute=0x20,
                race=0x08,
                attack=1800,
                defense=2600,
                lscale=6,
                rscale=6,
            ),
            name="D/D Savant Thomas",
        ),
    )
    assert info.typeline == ["Fiend", "Pendulum", "Effect"]
    assert info.scales is not None
    assert (info.scales.left, info.scales.right) == (6, 6)
    assert info.level_kind == "level"


def test_link_monster_has_arrows_and_no_defense():
    """Double Headed Anger Knuckle (146746, 0x4000021), link_marker 34 →
    BOTTOM|RIGHT. The row arrives with defense=None, and it must stay that way:
    a literal 0 would print "DEF/0" on a card that has no DEF."""
    info = build_card_info(
        146746,
        _FakeCardDB(
            _row(
                type=0x4000021,
                level=2,
                attribute=0x01,
                race=0x20,
                attack=1500,
                defense=None,
                link_marker=34,
            ),
            name="Double Headed Anger Knuckle",
        ),
    )
    assert info.typeline == ["Machine", "Link", "Effect"]
    assert info.level == 2
    assert info.level_kind == "link"
    assert info.defense is None
    assert info.link_arrows == ["RIGHT", "BOTTOM"]


def test_xyz_monster_level_kind_is_rank():
    info = build_card_info(
        1861629,
        _FakeCardDB(_row(type=0x800021, level=4), name="Xyz Test"),
    )
    assert info.level_kind == "rank"


def test_special_summon_bit_is_not_rendered():
    """Lava Golem (102380, 0x2000021) prints "Fiend / Effect". TYPE_SPSUMMON is
    an engine flag for nomi monsters, not a printed word."""
    info = build_card_info(
        102380,
        _FakeCardDB(
            _row(type=0x2000021, level=8, attribute=0x04, race=0x08, attack=3000, defense=2500),
            name="Lava Golem",
        ),
    )
    assert info.typeline == ["Fiend", "Effect"]


def test_token_suppresses_normal_but_keeps_other_labels():
    """Swordsoul Token (20001444, 0x5011) prints "Wyrm / Token / Tuner": the
    Normal body label is suppressed for tokens, but Tuner is not, and the
    stats cards.cdb records are still rendered."""
    info = build_card_info(
        20001444,
        _FakeCardDB(
            _row(type=0x5011, level=4, attribute=0x02, race=0x800000, attack=0, defense=0),
            name="Swordsoul Token",
        ),
    )
    assert info.typeline == ["Wyrm", "Token", "Tuner"]
    assert info.attribute == "WATER"
    assert info.level == 4
    assert info.attack == 0
    assert info.defense == 0


def test_quick_play_spell():
    """Parallel Teleport (483, 0x10002) — no attribute, level or ATK/DEF rows."""
    info = build_card_info(
        483,
        _FakeCardDB(
            _row(type=0x10002, level=0, attribute=0, race=0, attack=0, defense=0),
            name="Parallel Teleport",
        ),
    )
    assert info.card_type == "spell"
    assert info.typeline == ["Spell", "Quick-Play"]
    assert info.attribute is None
    assert info.race is None
    assert info.level is None
    assert info.level_kind is None
    assert info.attack is None
    assert info.defense is None


def test_counter_trap():
    """Rebound (983995, 0x100004)."""
    info = build_card_info(
        983995,
        _FakeCardDB(
            _row(type=0x100004, level=0, attribute=0, race=0, attack=0, defense=0),
            name="Rebound",
        ),
    )
    assert info.card_type == "trap"
    assert info.typeline == ["Trap", "Counter"]


def test_trap_carrying_monster_stats_reports_no_attribute_or_race():
    """Paleozoic Leanchoilia (1154611, 0x4) is a plain Trap whose race (Aqua) and
    attribute (WATER) columns are filled, because the card becomes a monster at
    runtime. Neither is printed on it."""
    info = build_card_info(
        1154611,
        _FakeCardDB(
            _row(type=0x4, level=0, attribute=0x02, race=0x40, attack=0, defense=0),
            name="Paleozoic Leanchoilia",
        ),
    )
    assert info.card_type == "trap"
    assert info.typeline == ["Trap"]
    assert info.attribute is None
    assert info.race is None


def test_ritual_spell_resolves_in_the_spell_context():
    """Revendread Evolution (7986397, 0x82). Bit 0x80 is "Ritual Monster" in the
    monster table and "Ritual Spell" here; resolving it against the wrong table
    would drop the word entirely."""
    info = build_card_info(
        7986397,
        _FakeCardDB(
            _row(type=0x82, level=0, attribute=0, race=0, attack=0, defense=0),
            name="Revendread Evolution",
        ),
    )
    assert info.card_type == "spell"
    assert info.typeline == ["Spell", "Ritual"]


def test_plain_spell_has_no_subtype_label():
    info = build_card_info(
        24094653,
        _FakeCardDB(
            _row(type=0x2, level=0, attribute=0, race=0, attack=0, defense=0),
            name="Polymerization",
        ),
    )
    assert info.typeline == ["Spell"]


def test_unknown_code_returns_none():
    class _Empty:
        def get_card(self, code):
            return None

    assert build_card_info(99999999, _Empty()) is None


def test_unrecognized_race_bit_is_omitted():
    """An unknown race must not render as hex — the label is dropped and the
    typeline still reads correctly."""
    info = build_card_info(1, _FakeCardDB(_row(race=1 << 50)))
    assert info.race is None
    assert info.typeline == ["Effect"]


def test_question_mark_atk_passes_through():
    """cards.cdb encodes "?" ATK/DEF as -2. The value is preserved;
    the client renders "?" for negatives."""
    info = build_card_info(10000, _FakeCardDB(_row(attack=-2, defense=-2)))
    assert info.attack == -2
    assert info.defense == -2


def test_missing_desc_becomes_empty_string():
    """get_card_desc returns None for a card with no text; the model's desc is
    non-optional, so it must be "" rather than null."""
    info = build_card_info(1, _FakeCardDB(_row(), desc=None))
    assert info.desc == ""


def test_structural_less_row_has_empty_typeline():
    """The two cdb junk rows (type == 0x4000) have no monster/spell/trap bit, so
    there is no table to resolve against."""
    info = build_card_info(
        10000110,
        _FakeCardDB(
            _row(type=0x4000, level=0, attribute=0, race=0, attack=0, defense=0),
            name="Unknown",
            desc=None,
        ),
    )
    assert info.card_type == "unknown"
    assert info.typeline == []
    assert info.desc == ""


# ─── Bit-partition guard (real cards.cdb) ───────────────────────────────────
# Every type bit on every card must be classified, IN ITS OWN CONTEXT: labelled
# for that card class, explicitly ignored, or structural. A union check would
# not do: TYPE_RITUAL (0x80) is in the monster table too, so dropping "Ritual"
# from SPELL_TRAP_TYPE_LABELS would still pass while every Ritual Spell silently
# rendered as bare ["Spell"].

from yugioh_core.constants import (
    IGNORED_TYPE_BITS,
    MONSTER_TYPE_LABELS,
    SPELL_TRAP_TYPE_LABELS,
    TYPE_MONSTER,
    TYPE_SPELL,
    TYPE_TRAP,
)

_STRUCTURAL = {TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP}


def test_every_type_bit_is_classified_in_its_context(cdb_column):
    monster_bits = {mask for mask, _ in MONSTER_TYPE_LABELS}
    spell_trap_bits = {mask for mask, _ in SPELL_TRAP_TYPE_LABELS}

    unclassified: dict[int, int] = {}
    for type_val in cdb_column("SELECT DISTINCT type FROM datas"):
        if type_val & TYPE_MONSTER:
            allowed = monster_bits | IGNORED_TYPE_BITS | _STRUCTURAL
        elif type_val & (TYPE_SPELL | TYPE_TRAP):
            allowed = spell_trap_bits | IGNORED_TYPE_BITS | _STRUCTURAL
        else:
            continue  # structural-less rows: see test_structural_less_rows_are_inert
        for bit_index in range(64):
            bit = 1 << bit_index
            if type_val & bit and bit not in allowed:
                unclassified[bit] = unclassified.get(bit, 0) + 1

    assert not unclassified, (
        "unclassified type bits: "
        + ", ".join(f"0x{bit:x} ({n} type values)" for bit, n in sorted(unclassified.items()))
        + " — add each to MONSTER_TYPE_LABELS / SPELL_TRAP_TYPE_LABELS or IGNORED_TYPE_BITS"
    )


def test_structural_less_rows_are_inert(card_db, cdb_column):
    """Rows with no monster/spell/trap bit are cdb detritus, which is why the
    classification sweep skips them. A real card would carry stats, so anything
    stat-bearing here means the sweep is skipping something it should check."""
    codes = cdb_column(
        "SELECT id FROM datas WHERE (type & ?) = 0",
        (TYPE_MONSTER | TYPE_SPELL | TYPE_TRAP,),
    )
    assert len(codes) < 20, f"{len(codes)} structural-less rows — no longer detritus"

    for code in codes:
        row = card_db.get_card(code)
        assert not any(row[field] for field in ("race", "attribute", "attack", "level")), (
            f"card {code} has stats but no structural bit — the sweep is skipping a real card"
        )

        info = build_card_info(code, card_db)
        assert info is not None
        assert info.card_type == "unknown"
        assert info.typeline == []


def test_no_spell_or_trap_reports_an_attribute_or_race(card_db, cdb_column):
    """Some spell/trap rows fill the race and attribute columns because those
    cards become monsters at runtime. Neither is printed, so neither may surface.

    Swept across every such row: a card's type bits don't reveal whether its row
    populates these columns, only the columns do."""
    codes = cdb_column(
        "SELECT id FROM datas WHERE (type & ?) != 0 AND (race != 0 OR attribute != 0)",
        (TYPE_SPELL | TYPE_TRAP,),
    )
    assert codes, "expected cards.cdb to contain spell/trap rows with stat columns set"

    offenders = []
    for code in codes:
        info = build_card_info(code, card_db)
        if info is not None and (info.attribute is not None or info.race is not None):
            offenders.append((code, info.name, info.card_type, info.attribute, info.race))
    assert not offenders, f"spell/trap cards surfacing monster stats: {offenders[:5]}"
