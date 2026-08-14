# tests/mud/fixtures/capture_mud_goldens.py
"""Freeze yugioh_mud's observation arrays.

Run deliberately, never automatically:
    .venv/bin/python tests/mud/fixtures/capture_mud_goldens.py

MUD keeps its own encoder but shares `encode_card`, `encode_u16` and
`encode_u32`, so a change to any shared primitive would move its arrays
silently.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
# tests/mud/fixtures/<this_file> → project root is three parents up.
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.mud.conftest import mud_observation_cases  # noqa: E402

if __name__ == "__main__":
    out: dict[str, np.ndarray] = {}
    for name, builder in mud_observation_cases().items():
        obs = builder()
        for field in ("cards", "global_state", "actions", "action_mask"):
            out[f"{name}_{field}"] = np.asarray(obs[field])
    np.savez_compressed(HERE / "mud_goldens.npz", **out)
    print(f"wrote mud_goldens.npz ({len(out) // 4} cases)")
