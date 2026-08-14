"""Check `encode_observation` against the committed goldens, byte for byte.

The goldens are written by the capture script beside them; these tests only
read. What a match proves differs by entry: the per-prompt goldens were
recorded from a separate encoder, so matching them shows the two agree, while
the route-case goldens came from `encode_observation` and pin only what it
already does.

Coverage is enumerated, not sampled. Random duels miss the byte-10 overlay
branch on most decks, and a field-by-field audit of this same problem found
`subsequence` while missing `position`. One case per descriptor kind makes a gap
a missing parameter rather than a missing thought.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tests.env.conftest import MINIMAL_MSGS, ROUTE_CASES, obs_from_msg
from yugioh_rl.obs_encoder import encode_observation

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "env" / "fixtures"
ACTION_GOLDENS = json.loads((FIXTURES / "encoder_goldens_actions.json").read_text())


def _assert_actions_match(case: str) -> np.ndarray:
    """Compare against the frozen goldens and return the new bytes for further
    assertions on a specific route.

    A `KeyError` here means a prompt was added to `MINIMAL_MSGS` or
    `ROUTE_CASES` without recapturing the fixture, which would leave the new
    case unpinned.
    """
    golden = ACTION_GOLDENS[case]
    obs = obs_from_msg(golden["msg"])
    encoded = encode_observation(obs)
    np.testing.assert_array_equal(encoded["actions"], np.array(golden["actions"], dtype=np.uint8))
    np.testing.assert_array_equal(encoded["action_mask"], np.array(golden["mask"], dtype=np.int8))
    # The count is frozen alongside the bytes, so the dense list's length is
    # pinned too, not just its contents.
    assert obs.num_actions == golden["num_actions"]
    return encoded["actions"]


@pytest.mark.parametrize("msg_type", sorted(MINIMAL_MSGS))
def test_actions_match_the_frozen_goldens(msg_type: int) -> None:
    _assert_actions_match(str(msg_type))


# Per-kind cases can still leave a byte silently zero, because three bytes
# carry a position-shaped value by different routes. The prompts live in
# `ROUTE_CASES` so the capture script freezes the very bytes asserted here.


def _route_actions(case: str) -> np.ndarray:
    assert case in ROUTE_CASES  # the golden and the shared prompt are one case
    return _assert_actions_match(case)


def test_byte10_position_branch() -> None:
    assert int(_route_actions("byte10_position_branch")[0][10]) == 0x5


def test_byte10_overlay_branch() -> None:
    """LOCATION_OVERLAY set, so the slot is a stack index, not a position."""
    assert int(_route_actions("byte10_overlay_branch")[0][10]) == 2


def test_byte11_chain_route() -> None:
    """The chain extractor is byte 11's only producer."""
    assert int(_route_actions("byte11_chain_route")[0][11]) == 0x5


def test_byte16_choose_position_route() -> None:
    """_extract_position_actions puts its bitmask in `index`, so it lands in
    byte 16 -- not byte 11, despite being a position."""
    actions = _route_actions("byte16_choose_position_route")
    assert {int(actions[0][16]), int(actions[1][16])} == {0x1, 0x4}
    assert int(actions[0][11]) == 0


def test_byte12_direct_attackable_route() -> None:
    """`MINIMAL_MSGS`'s attackable card always carries `direct_attackable: 0`,
    so the per-kind case never exercises the byte turning on."""
    assert int(_route_actions("byte12_direct_attackable_route")[0][12]) == 1


def test_matches_the_frozen_observation_goldens(lib, db_path, script_dirs, assets_dir) -> None:
    """Real duels across all eight decks, replayed against a frozen encoder-output capture.

    Replay depth comes from the fixture, not a literal: the capture plays the
    Xyz decks far deeper than the rest to reach the rows where byte 10 carries
    an overlay stack index, and a shallower loop here would leave exactly
    those rows uncompared.

    The capture pins the engine and CardScripts as much as the encoder, so a
    `cards` mismatch in the opening steps points at those having moved --
    recapture the fixture rather than hunting the encoder.
    """
    from yugioh_env.deck_parser import parse_ydk
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.opponent import RandomOpponent
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    goldens = np.load(FIXTURES / "encoder_goldens_observations.npz")
    depth: dict[str, int] = {}
    for key in goldens.files:
        if key.endswith("_cards"):
            deck, step = key[: -len("_cards")].rsplit("_", 1)
            depth[deck] = max(depth.get(deck, 0), int(step) + 1)
    env = YuGiOhEnvironment({})
    env.set_opponent(RandomOpponent(seed=0))
    checked = 0
    try:
        for deck_path in sorted((assets_dir / "decks").glob("*.ydk")):
            deck = parse_ydk(str(deck_path))
            name = deck_path.stem
            obs = env.reset(seed=7, deck0=deck, deck1=deck, agent_player=0)
            rng = np.random.default_rng(7)
            for step in range(depth.get(name, 0)):
                if obs.done or f"{name}_{step}_cards" not in goldens.files:
                    break
                encoded = encode_observation(obs)
                for field, suffix in (
                    ("cards", "cards"),
                    ("global_state", "global"),
                    ("actions", "actions"),
                    ("action_mask", "mask"),
                ):
                    np.testing.assert_array_equal(
                        encoded[field],
                        goldens[f"{name}_{step}_{suffix}"],
                        err_msg=f"{name} step {step} {field}",
                    )
                checked += 1
                if obs.num_actions == 0:
                    break
                # Same range the capture script draws from, so the replay
                # follows the fixture's action sequence.
                obs = env.step(YuGiOhAction(action_index=int(rng.choice(obs.num_actions))))
    finally:
        env.close()
    assert checked == sum(depth.values()), (
        f"compared {checked} of {sum(depth.values())} captured observations -- the replay "
        f"diverged from the capture, or the fixture holds rows this test cannot reach"
    )
