"""Unit tests for the MUD card name lookup.

Uses a temporary SQLite database — no cards.cdb required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from yugioh_mud.card_lookup import CardNameLookup


@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal cards.cdb with texts + datas tables."""
    db_path = tmp_path_factory.mktemp("cards") / "cards.cdb"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE texts ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT,"
        "  desc TEXT"
        ")"
    )
    conn.execute(
        "CREATE TABLE datas ("
        "  id INTEGER PRIMARY KEY,"
        "  ot INTEGER, alias INTEGER, setcode INTEGER,"
        "  type INTEGER, atk INTEGER, def INTEGER,"
        "  level INTEGER, race INTEGER, attribute INTEGER"
        ")"
    )
    conn.executemany(
        "INSERT INTO texts (id, name, desc) VALUES (?, ?, ?)",
        [
            (89631139, "Blue-Eyes White Dragon", "desc1"),
            (89631140, "Blue-Eyes White Dragon", "desc1"),  # alt artwork
            (46986414, "Dark Magician", "desc2"),
            (40640057, "Kuriboh", "desc3"),
            (99999999, "", "empty name"),
        ],
    )
    conn.executemany(
        "INSERT INTO datas (id, alias) VALUES (?, ?)",
        [
            (89631139, 0),          # canonical
            (89631140, 89631139),   # alt artwork → points to canonical
            (46986414, 0),
            (40640057, 0),
            (99999999, 0),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture(scope="module")
def lookup(tmp_db: Path) -> CardNameLookup:
    """Single CardNameLookup shared across all tests in the module."""
    return CardNameLookup(tmp_db)


class TestCardNameLookup:
    def test_name_to_code(self, lookup: CardNameLookup):
        assert lookup.name_to_code("Blue-Eyes White Dragon") == 89631139
        assert lookup.name_to_code("Dark Magician") == 46986414
        assert lookup.name_to_code("Kuriboh") == 40640057

    def test_unknown_name_returns_none(self, lookup: CardNameLookup):
        assert lookup.name_to_code("Nonexistent Card") is None

    def test_empty_name_not_indexed(self, lookup: CardNameLookup):
        assert lookup.name_to_code("") is None

    def test_len(self, lookup: CardNameLookup):
        # 3 cards with non-empty names
        assert len(lookup) == 3

    def test_contains(self, lookup: CardNameLookup):
        assert "Dark Magician" in lookup
        assert "Nonexistent" not in lookup

    def test_alternate_artwork_returns_canonical(self, lookup: CardNameLookup):
        """When multiple rows share a name, the canonical (alias=0) wins."""
        # 89631139 is canonical (alias=0), 89631140 is alt (alias=89631139)
        assert lookup.name_to_code("Blue-Eyes White Dragon") == 89631139
