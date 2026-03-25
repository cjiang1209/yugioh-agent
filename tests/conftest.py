"""Shared test fixtures."""

from __future__ import annotations

import os
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
    return assets_dir / "decks" / "starter.ydk"


@pytest.fixture
def db_path(assets_dir) -> Path:
    path = assets_dir / "cards.cdb"
    if not path.exists():
        pytest.skip("cards.cdb not found. Download it to assets/cards.cdb")
    return path


@pytest.fixture
def script_dirs(project_root) -> list[Path]:
    dirs = [
        project_root / "third_party" / "CardScripts" / "official",
        project_root / "third_party" / "CardScripts" / "pre-release",
        project_root / "third_party" / "CardScripts",
    ]
    existing = [d for d in dirs if d.exists()]
    if not existing:
        pytest.skip("CardScripts not found. Set up git submodules.")
    return existing


@pytest.fixture
def lib():
    """Load the OCG core library."""
    try:
        from yugioh_env.lib_loader import load_library
        return load_library()
    except FileNotFoundError:
        pytest.skip("libocgcore not found. Run: make build")


@pytest.fixture
def card_db(db_path):
    """Create a CardDatabase instance."""
    from yugioh_core.card_database import CardDatabase
    db = CardDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def duel(lib, card_db, script_dirs, deck_path):
    """Create a Duel instance ready to use."""
    from yugioh_env.duel import Duel
    d = Duel(lib, card_db, script_dirs)
    yield d
    d.destroy()
