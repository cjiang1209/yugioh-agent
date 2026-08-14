"""Shared fixtures and markers for RL tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

requires_engine = pytest.mark.skipif(
    not (Path("build/libocgcore.dylib").exists() or Path("build/libocgcore.so").exists())
    or not Path("assets/cards.cdb").exists(),
    reason="libocgcore.{dylib,so} (run `make build`) and assets/cards.cdb required",
)


@pytest.fixture
def lib():
    """Load the OCG core library.

    Duplicated from tests/env/conftest.py: pytest fixtures don't cross
    directory boundaries, so tests here that drive a live duel need their
    own copy.
    """
    try:
        from yugioh_env.lib_loader import load_library

        return load_library()
    except FileNotFoundError:
        pytest.skip("libocgcore not found. Run: make build")


@pytest.fixture
def script_dirs(project_root):
    """Duplicated from tests/env/conftest.py; see `lib` above."""
    dirs = [
        project_root / "third_party" / "CardScripts" / "official",
        project_root / "third_party" / "CardScripts" / "pre-release",
        project_root / "third_party" / "CardScripts",
    ]
    existing = [d for d in dirs if d.exists()]
    if not existing:
        pytest.skip("CardScripts not found. Set up git submodules.")
    return existing


def hash_obs_field(arr: np.ndarray) -> str:
    """Stable hex digest of an observation array's bytes.

    Used by both the baseline-capture script and the bit-equality
    regression test so the two stay in sync.
    """
    return hashlib.sha1(arr.tobytes()).hexdigest()
