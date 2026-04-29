"""Shared test fixtures."""

from __future__ import annotations

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
