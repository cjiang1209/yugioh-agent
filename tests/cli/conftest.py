"""Shared fixtures for cli/ tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def deck_path_str(deck_path: Path) -> str:
    """Absolute starter-deck path as a string (subprocess args + argparse-friendly)."""
    return str(deck_path)
