"""Tests for StringResolver."""

import sqlite3
from pathlib import Path

import pytest

from yugioh_core.card_database import CardDatabase
from yugioh_core.string_resolver import StringResolver


@pytest.fixture
def db(tmp_path: Path) -> CardDatabase:
    """Build a tiny cards.cdb-shaped DB with one card and three filled str slots."""
    path = tmp_path / "test.cdb"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE datas(id integer primary key, ot int, alias int, setcode int,
                           type int, atk int, def int, level int, race int, attribute int);
        CREATE TABLE texts(id integer primary key, name text, desc text,
                           str1 text, str2 text, str3 text, str4 text, str5 text,
                           str6 text, str7 text, str8 text, str9 text, str10 text,
                           str11 text, str12 text, str13 text, str14 text,
                           str15 text, str16 text);
        INSERT INTO datas VALUES (10032958, 0, 0, 0, 33, 0, 0, 0, 0, 0);
        INSERT INTO texts(id, name, str1, str2, str3) VALUES
            (10032958, 'Divine Dragon - Excelion',
             'Gain an effect', 'Increase ATK by 1000', 'Make 1 additional attack');
        """
    )
    conn.commit()
    conn.close()
    return CardDatabase(path)


def _make_stringid(passcode: int, n: int) -> int:
    return (passcode << 20) | (n & 0xfffff)


def test_per_card_resolution_returns_str_slot(db):
    """A per-card desc decodes to (passcode, n) and returns texts.str{n+1}."""
    r = StringResolver(db)
    assert r.resolve(_make_stringid(10032958, 0)) == "Gain an effect"
    assert r.resolve(_make_stringid(10032958, 1)) == "Increase ATK by 1000"
    assert r.resolve(_make_stringid(10032958, 2)) == "Make 1 additional attack"


def test_unknown_card_returns_none(db):
    """A passcode not in cards.cdb returns None — caller falls back to placeholder."""
    r = StringResolver(db)
    assert r.resolve(_make_stringid(99999999, 0)) is None


def test_empty_str_slot_returns_none(db):
    """Slot 4 is unset on the test card; lookup returns None, not the empty string,
    so the describer's `if resolved` fallback fires correctly."""
    r = StringResolver(db)
    assert r.resolve(_make_stringid(10032958, 3)) is None


def test_sysstring_returns_none_when_no_table(db):
    """Sysstrings (passcode==0) are out of scope until a sysstring table is wired in.
    Until then, they must return None so the placeholder fallback fires."""
    r = StringResolver(db)
    assert r.resolve(0x46) is None  # what we saw as 'effect 0x46' in smoke logs


def test_sysstring_table_used_when_provided(db):
    """When a sysstring table is wired in (future), passcode==0 IDs resolve from it."""
    sys = {0x46: "Monster Cards", 0x47: "Spell Cards", 0x48: "Trap Cards"}
    r = StringResolver(db, sys_strings=sys)
    assert r.resolve(0x46) == "Monster Cards"
    assert r.resolve(0x47) == "Spell Cards"
    # Unknown sysstring still falls through to None.
    assert r.resolve(0x99) is None


def test_per_card_lookup_ignores_sys_table(db):
    """Per-card desc (passcode > 0) ignores the sysstring table — they're disjoint."""
    sys = {0: "WRONG"}  # would be picked up if sys table were consulted for n==0
    r = StringResolver(db, sys_strings=sys)
    assert r.resolve(_make_stringid(10032958, 0)) == "Gain an effect"
