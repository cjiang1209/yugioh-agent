# tests/env/fixtures/capture_encoder_goldens.py
"""Write the golden fixtures, which the encoder tests then check against.

Run deliberately, never automatically:
    .venv/bin/python tests/env/fixtures/capture_encoder_goldens.py

Producing the goldens and checking them are separate jobs: this script writes
the files, and the tests only read them. So a rerun rewrites the oracle -- do
it to add a case, and confirm the existing entries come back unchanged.

The per-prompt entries were recorded from a separate encoder, absent from this
tree, so an encoder that reproduces them agrees with one nothing here can run.
The route-case entries were recorded from the encoder in this tree, so they
pin what it already does and nothing more.
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

from tests.env.conftest import MINIMAL_MSGS, ROUTE_CASES, obs_from_msg  # noqa: E402

from yugioh_rl.obs_encoder import encode_observation  # noqa: E402

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
    """One entry per registered msg_type, plus one per named route case.

    msg_type entries are keyed by the number and route cases by their name --
    two route cases share a msg_type, so the number cannot key them. Each
    entry's stored `msg` carries its own `msg_type`, so a reader replays a
    prompt without having to parse the key.
    """
    out: dict[str, dict] = {}
    cases = [(str(mt), {**msg, "msg_type": mt}) for mt, msg in sorted(MINIMAL_MSGS.items())]
    cases += [(name, msg) for name, msg in sorted(ROUTE_CASES.items())]
    for key, msg in cases:
        obs = obs_from_msg(msg)
        encoded = encode_observation(obs)
        out[key] = {
            "msg": dict(msg),
            "actions": encoded["actions"].tolist(),
            "mask": encoded["action_mask"].tolist(),
            "num_actions": obs.num_actions,
        }
    return out


def capture_observations() -> dict[str, np.ndarray]:
    """Real-duel observations across every bundled deck.

    All eight decks, run to different depths. The four decks in `deep`
    run `_DEEP_STEPS`: reaching an action whose byte 7 carries
    LOCATION_OVERLAY -- and so byte 10 holding an overlay stack index instead
    of a position bitmask -- needs an Xyz Summon to land under random legal
    play and then a later prompt over that monster's materials, and even at
    `_DEEP_STEPS` it is a rare event. The remaining four decks have no Xyz
    monsters at all, so no amount of extra depth can ever produce that row;
    they run only `_SHALLOW_STEPS`.
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
                if obs.done or obs.num_actions == 0:
                    break
                key = f"{name}_{step}"
                encoded = encode_observation(obs)
                out[f"{key}_cards"] = encoded["cards"]
                out[f"{key}_global"] = encoded["global_state"]
                out[f"{key}_actions"] = encoded["actions"]
                out[f"{key}_mask"] = encoded["action_mask"]
                obs = env.step(YuGiOhAction(action_index=int(rng.choice(obs.num_actions))))
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
