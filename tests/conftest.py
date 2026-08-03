"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    return _project_root()


@pytest.fixture
def assets_dir(project_root) -> Path:
    return project_root / "assets"


@pytest.fixture
def deck_path(assets_dir) -> Path:
    return assets_dir / "decks" / "blue_eyes.ydk"


@pytest.fixture
def db_path(assets_dir) -> Path:
    path = assets_dir / "cards.cdb"
    if not path.exists():
        pytest.skip("cards.cdb not found. Download it to assets/cards.cdb")
    return path


@pytest.fixture
def card_db(db_path):
    """Create a CardDatabase instance."""
    from yugioh_core.card_database import CardDatabase

    db = CardDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def cdb_column(db_path):
    """Run a single-column query against cards.cdb and return the values.

    Use this to *discover* a card exhibiting the property under test rather than
    hardcoding a passcode: cards.cdb is upstream data that gains, loses and
    rewrites rows, so a pinned example is a scheduled failure.
    """

    def query(sql: str, params: tuple = ()) -> list:
        conn = sqlite3.connect(db_path)
        try:
            return [value for (value,) in conn.execute(sql, params)]
        finally:
            conn.close()

    return query
