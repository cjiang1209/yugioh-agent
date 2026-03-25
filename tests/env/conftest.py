"""Env-specific test fixtures (lib, script_dirs, duel)."""

from __future__ import annotations

from pathlib import Path

import pytest


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
def duel(lib, card_db, script_dirs, deck_path):
    """Create a Duel instance ready to use."""
    from yugioh_env.duel import Duel
    d = Duel(lib, card_db, script_dirs)
    yield d
    d.destroy()
