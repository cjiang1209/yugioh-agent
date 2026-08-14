# tests/env/fixtures/capture_encoder_goldens.py
"""Freeze the current encoder's output as the equivalence oracle.

Run deliberately, never automatically:
    .venv/bin/python tests/env/fixtures/capture_encoder_goldens.py

These goldens pin the bytes the network receives. Any encoder checked against
them must reproduce them exactly; a diff is a regression until proven otherwise.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
# tests/env/fixtures/<this_file> → project root is three parents up.
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.env.conftest import MINIMAL_MSGS  # noqa: E402

from yugioh_env.action_space import ActionMapper  # noqa: E402

DECKS = sorted(ROOT.glob("assets/decks/*.ydk"))

_DEEP_STEPS = 400
_SHALLOW_STEPS = 40


def _xyz_decks() -> set[str]:
    """Decks holding at least one Xyz monster.

    Only these can land an Xyz Summon and then offer a selection from that
    monster's attached materials, which is what puts LOCATION_OVERLAY in an
    action's byte 7 and, with it, an overlay stack index rather than a
    position bitmask in byte 10. The rest cannot contribute such a row at any
    depth, so they run shallow.

    Read from the card database rather than named, so adding an Xyz deck to
    assets/decks does not silently leave it too shallow to reach that row.
    """
    import sqlite3

    from yugioh_core.constants import TYPE_XYZ
    from yugioh_env.deck_parser import parse_ydk

    out: set[str] = set()
    with sqlite3.connect(ROOT / "assets" / "cards.cdb") as con:
        for deck_path in DECKS:
            deck = parse_ydk(str(deck_path))
            codes = list(deck.get("main", [])) + list(deck.get("extra", []))
            if not codes:
                continue
            placeholders = ",".join("?" * len(codes))
            row = con.execute(
                f"SELECT COUNT(*) FROM datas WHERE id IN ({placeholders}) AND type & ?",
                (*codes, TYPE_XYZ),
            ).fetchone()
            if row[0]:
                out.add(deck_path.stem)
    return out


def capture_actions() -> dict:
    """One entry per registered msg_type, driven by MINIMAL_MSGS."""
    out: dict[str, dict] = {}
    for msg_type, msg in sorted(MINIMAL_MSGS.items()):
        mapper = ActionMapper()
        mapper.update({**msg, "msg_type": msg_type, "_agent_player": 0})
        out[str(msg_type)] = {
            "msg": dict(msg),
            "actions": mapper.get_action_features().tolist(),
            "mask": mapper.get_action_mask().tolist(),
            "num_actions": mapper.num_actions,
        }
    return out


def capture_observations() -> dict[str, np.ndarray]:
    """Real-duel observations across every bundled deck, at one of two depths.

    The deep depth exists for the overlay row `_xyz_decks` describes: it needs
    an Xyz Summon to land under random legal play and then a later prompt over
    that monster's materials, which stays rare even at that depth.
    """
    from yugioh_env.deck_parser import parse_ydk
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.opponent import RandomOpponent
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    deep = _xyz_decks()
    out: dict[str, np.ndarray] = {}
    env = YuGiOhEnvironment({})
    env.set_opponent(RandomOpponent(seed=0))
    try:
        for deck_path in DECKS:
            deck = parse_ydk(str(deck_path))
            name = deck_path.stem
            steps = _DEEP_STEPS if name in deep else _SHALLOW_STEPS
            obs = env.reset(seed=7, deck0=deck, deck1=deck, agent_player=0)
            rng = np.random.default_rng(7)
            for step in range(steps):
                if obs.done:
                    break
                legal = np.flatnonzero(np.asarray(obs.action_mask) == 1)
                if legal.size == 0:
                    break
                key = f"{name}_{step}"
                out[f"{key}_cards"] = np.asarray(obs.cards)
                out[f"{key}_global"] = np.asarray(obs.global_state)
                out[f"{key}_actions"] = np.asarray(obs.actions)
                out[f"{key}_mask"] = np.asarray(obs.action_mask)
                obs = env.step(YuGiOhAction(action_index=int(rng.choice(legal))))
    finally:
        env.close()
    return out


if __name__ == "__main__":
    (HERE / "encoder_goldens_actions.json").write_text(
        json.dumps(capture_actions(), indent=2, sort_keys=True) + "\n"
    )
    print("wrote encoder_goldens_actions.json")
    obs_goldens = capture_observations()
    np.savez_compressed(HERE / "encoder_goldens_observations.npz", **obs_goldens)
    print(f"wrote encoder_goldens_observations.npz ({len(obs_goldens) // 4} observations)")
