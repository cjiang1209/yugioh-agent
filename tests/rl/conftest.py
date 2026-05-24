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


def hash_obs_field(arr: np.ndarray) -> str:
    """Stable hex digest of an observation array's bytes.

    Used by both the baseline-capture script and the bit-equality
    regression test so the two stay in sync.
    """
    return hashlib.sha1(arr.tobytes()).hexdigest()
