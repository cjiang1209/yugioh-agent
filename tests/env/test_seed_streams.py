"""Nothing in a duel's setup draws from the per-duel seed itself.

Two generators built from one seed track each other: `Random(s).randint(...)`
and `Random(s).shuffle(...)` read the same first draws, so the result of one
predicts the other. A seat drawn from the deck-order stream is correlated with
the deck order, which tilts who wins. `duel_seed` is a master seed; every
consumer takes its own derived stream.

This records the seed behind every generator built during `reset()` rather than
naming them individually -- a consumer added later is covered without anyone
remembering to extend the test.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from env.server.environment import YuGiOhEnvironment

DUEL_SEED = 12345


def _config(db_path, script_dirs, deck_path) -> dict:
    return {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
    }


def _seeds_drawn_during_reset(env) -> list[int]:
    """Every seed a `random.Random` is constructed from during one reset.

    Patching the class catches the seat draw in the environment and the shuffle
    inside `Duel` alike, so no consumer can be missed by naming the wrong one.
    """
    with patch("random.Random", wraps=random.Random) as spy:
        env.reset(seed=DUEL_SEED, agent_player="random")
    seeds = [call.args[0] if call.args else call.kwargs.get("x") for call in spy.call_args_list]
    return [seed for seed in seeds if seed is not None]


@pytest.fixture
def drawn(lib, db_path, script_dirs, deck_path):
    env = YuGiOhEnvironment(_config(db_path, script_dirs, deck_path))
    try:
        yield _seeds_drawn_during_reset(env)
    finally:
        env.close()


def test_reset_builds_several_generators(drawn):
    """Guards the two below: they prove nothing if nothing was captured."""
    assert len(drawn) >= 3, (
        f"expected reset() to build a generator per consumer, captured {len(drawn)}"
    )


def test_no_generator_is_built_from_the_master_seed(drawn):
    assert DUEL_SEED not in drawn, (
        f"a generator was seeded with the master seed {DUEL_SEED}, so its draws "
        f"track every other stream derived from it; seeds drawn were {drawn}"
    )


def test_generators_do_not_share_a_seed(drawn):
    duplicates = {s for s in drawn if drawn.count(s) > 1}
    assert not duplicates, (
        f"generators share seeds {sorted(duplicates)}, so their draws are "
        f"correlated; seeds drawn were {drawn}"
    )
