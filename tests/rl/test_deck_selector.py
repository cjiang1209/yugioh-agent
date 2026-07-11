import random

from yugioh_rl.deck_selector import DeckSelector


def _legacy_random(seed, episode_idx, n):
    rng = random.Random(seed + episode_idx)
    return rng.randrange(n), rng.randrange(n)


def test_random_is_deterministic_and_matches_seed_stream():
    s = DeckSelector(pool_size=31, seed=42, allocation="random")
    assert s.select(5) == s.select(5)
    assert s.select(5) == _legacy_random(42, 5, 31)


def test_balanced_agent_is_round_robin():
    s = DeckSelector(pool_size=31, seed=42, allocation="balanced")
    assert s.select(1)[0] == 0
    assert s.select(2)[0] == 1
    assert s.select(31)[0] == 30
    assert s.select(32)[0] == 0  # wraps


def test_balanced_covers_all_decks_uniformly():
    n = 31
    s = DeckSelector(pool_size=n, seed=42, allocation="balanced")
    counts = [0] * n
    for ep in range(1, 101):
        counts[s.select(ep)[0]] += 1
    assert min(counts) >= 3 and max(counts) <= 4
    assert all(c > 0 for c in counts)


def test_mirror_forces_equal_indices_in_both_modes():
    for alloc in ("random", "balanced"):
        s = DeckSelector(pool_size=31, seed=42, allocation=alloc, mirror=True)
        for ep in (1, 7, 40):
            a, o = s.select(ep)
            assert a == o


def test_non_mirror_opponent_can_differ():
    s = DeckSelector(pool_size=31, seed=42, allocation="balanced", mirror=False)
    assert any(s.select(ep)[0] != s.select(ep)[1] for ep in range(1, 40))
