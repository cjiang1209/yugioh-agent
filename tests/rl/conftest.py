"""Shared fixtures and markers for RL tests."""
from __future__ import annotations

from pathlib import Path

import pytest


requires_engine = pytest.mark.skipif(
    not (Path("build/libocgcore.dylib").exists()
         or Path("build/libocgcore.so").exists())
    or not Path("assets/cards.cdb").exists(),
    reason="libocgcore.{dylib,so} (run `make build`) and assets/cards.cdb required",
)
