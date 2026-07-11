"""Balanced eval gives uniform per-deck coverage + worker determinism.
Skips without torch/libocgcore/cdb."""

import pytest

torch = pytest.importorskip("torch")


def _pool(assets_dir, n=4):
    from yugioh_rl.env_wrapper import parse_deck_pool

    names = ["blue_eyes", "dark_magician", "hero", "utopia"][:n]
    return parse_deck_pool([str(assets_dir / "decks" / f"{x}.ydk") for x in names])


def test_balanced_uniform_coverage(lib, db_path, script_dirs, assets_dir):
    from yugioh_rl.eval import evaluate

    pool = _pool(assets_dir, 4)
    res = evaluate(
        agent_spec="random",
        deck_pool=pool,
        opponent_specs=["random"],
        num_episodes=8,
        seed=0,
        workers=1,
        deck_allocation="balanced",
    )[0]
    counts = {k: len(v) for k, v in res.per_deck_wins.items()}
    # 8 episodes / 4 decks → exactly 2 each, all decks present
    assert sorted(counts.values()) == [2, 2, 2, 2]


def test_worker_determinism_preserved(lib, db_path, script_dirs, assets_dir):
    from yugioh_rl.eval import evaluate

    pool = _pool(assets_dir, 4)
    kw = dict(
        agent_spec="random",
        deck_pool=pool,
        opponent_specs=["random"],
        num_episodes=8,
        seed=0,
        deck_allocation="balanced",
    )
    r1 = evaluate(**kw, workers=1)[0]
    r2 = evaluate(**kw, workers=2)[0]
    assert r1.wins == r2.wins and r1.episodes == r2.episodes
    assert {k: sorted(v) for k, v in r1.per_deck_wins.items()} == {
        k: sorted(v) for k, v in r2.per_deck_wins.items()
    }
