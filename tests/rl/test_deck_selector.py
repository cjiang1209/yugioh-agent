import random

from core.seeding import SEAT, substream
from rl.deck_selector import DeckSelector


def test_random_selection_is_a_pure_function_of_seed_and_episode():
    s = DeckSelector(pool_size=31, seed=42, allocation="random")
    assert s.select(5) == s.select(5)
    assert DeckSelector(pool_size=31, seed=42, allocation="random").select(5) == s.select(5)
    assert DeckSelector(pool_size=31, seed=43, allocation="random").select(5) != s.select(5)


def test_deck_choice_does_not_track_turn_order():
    """The deck a side plays must not predict which seat it takes.

    Turn order is worth about 15pp here, so a deck whose seat is predictable
    folds that advantage into its win rate and the number stops measuring the
    deck. Drawing both from one seed makes the seat exactly predictable.
    """
    for pool_size in (2, 8):
        s = DeckSelector(pool_size=pool_size, seed=42, allocation="random")
        first = [0] * pool_size
        total = [0] * pool_size
        for episode in range(1, 4001):
            deck, _ = s.select(episode)
            seat = random.Random(substream(42 + episode, SEAT)).randint(0, 1)
            total[deck] += 1
            first[deck] += seat == 0
        for deck in range(pool_size):
            share = first[deck] / total[deck]
            assert 0.4 < share < 0.6, (
                f"pool={pool_size}: deck {deck} goes first {share:.0%} of the time, "
                f"so its seat is predictable from the deck"
            )


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
