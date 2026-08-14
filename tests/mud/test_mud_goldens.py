# tests/mud/test_mud_goldens.py
"""MUD's arrays must not move. It keeps its own encoder but shares the
encode_* primitives, so this is the guard behind "MUD untouched"."""

from __future__ import annotations

import pathlib

import numpy as np

from tests.mud.conftest import mud_observation_cases

GOLDENS = np.load(pathlib.Path(__file__).parent / "fixtures" / "mud_goldens.npz")


def test_mud_observation_arrays_unchanged() -> None:
    for name, builder in mud_observation_cases().items():
        obs = builder()
        for field in ("cards", "global_state", "actions", "action_mask"):
            np.testing.assert_array_equal(
                np.asarray(obs[field]),
                GOLDENS[f"{name}_{field}"],
                err_msg=f"MUD {name}.{field} moved -- a shared encode_* primitive changed",
            )
