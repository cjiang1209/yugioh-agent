"""Streams derived from one master seed must not predict each other."""

from __future__ import annotations

import random

from core.seeding import (
    _EPISODE_NS,
    DECK_CHOICE,
    DECK_ORDER,
    ENGINE,
    OPPONENT,
    RECOMMENDER,
    SEAT,
    SNAPSHOT_CHOICE,
    episode_master_seed,
    state_words,
    substream,
)

STREAMS = (SEAT, OPPONENT, DECK_ORDER, ENGINE)
MASTERS = range(1, 4001)


def test_substream_is_a_pure_function():
    assert substream(42, SEAT) == substream(42, SEAT)


def test_streams_are_distinct_for_every_master():
    collisions = [m for m in MASTERS if len({substream(m, s) for s in STREAMS}) != len(STREAMS)]
    assert not collisions, f"streams collided for masters {collisions[:5]}"


def test_no_stream_returns_the_master_unchanged():
    """A pass-through derivation would reintroduce the defect silently."""
    offenders = [(m, s) for m in MASTERS for s in STREAMS if substream(m, s) == m]
    assert not offenders, f"substream returned the master seed for {offenders[:5]}"


_DECK_SIZE = 40
# Chi-square over a 2 x 40 table, so 39 degrees of freedom. Independence sits
# near 39 and p=0.001 is about 72. Measured: a sound derivation scores ~35, and
# routing both draws through one stream scores ~3200. Anywhere between those is
# a threshold; 120 leaves both sides an order of magnitude of room.
_CHI2_LIMIT = 120.0


def _chi2_seat_vs_deck(derive) -> float:
    """Association between the seat draw and the deck's first shuffle draw."""
    deck = list(range(_DECK_SIZE))
    observed: dict[tuple[int, int], int] = {}
    seat_totals = [0, 0]
    card_totals = [0] * _DECK_SIZE
    for master in MASTERS:
        seat = random.Random(derive(master, SEAT)).randint(0, 1)
        shuffled = deck[:]
        random.Random(derive(master, DECK_ORDER)).shuffle(shuffled)
        # Fisher-Yates walks downward, so the first draw lands the LAST position.
        # Reading deck[0] instead almost misses the effect.
        card = shuffled[-1]
        observed[(seat, card)] = observed.get((seat, card), 0) + 1
        seat_totals[seat] += 1
        card_totals[card] += 1

    total = len(MASTERS)
    stat = 0.0
    for seat in (0, 1):
        for card in range(_DECK_SIZE):
            expected = seat_totals[seat] * card_totals[card] / total
            if expected:
                stat += (observed.get((seat, card), 0) - expected) ** 2 / expected
    return stat


def test_seat_draw_does_not_track_the_deck_shuffle():
    """The defect this module exists to prevent, stated as a measurement.

    A seat and a shuffle drawn from one seed read the same first draws, so the
    seat predicts the deck order -- and the deck order tilts who wins.
    """
    stat = _chi2_seat_vs_deck(substream)
    assert stat < _CHI2_LIMIT, (
        f"seat/deck chi-square is {stat:.1f} against a limit of {_CHI2_LIMIT}; "
        f"the seat draw is tracking the deck stream"
    )


def test_the_measurement_catches_a_shared_stream():
    """Guards the test above: it is only meaningful if it can go red.

    A derivation that hands every caller the master seed is the original bug.
    """
    stat = _chi2_seat_vs_deck(lambda master, _stream: master)
    assert stat > _CHI2_LIMIT, (
        f"a shared stream scored only {stat:.1f}, under the {_CHI2_LIMIT} limit -- "
        f"the independence test above would not catch the defect it exists for"
    )


def test_episodes_do_not_collide_across_parallel_runs():
    """Workers are spaced apart in run seed, and they outrun the spacing.

    A subproc worker takes run seed ``base + i * 10000`` and a long run reaches
    roughly 34k episodes per worker, so numbering episodes by addition puts
    worker 0's episode 10001 on worker 1's episode 1: the same deck, seat and
    engine RNG, so the workers duplicate each other's experience.
    """
    spacing = 10_000
    episodes = 35_000
    seen: dict[int, tuple[int, int]] = {}
    for worker in range(4):
        run_seed = 42 + worker * spacing
        for episode in (1, 2, spacing + 1, episodes):
            master = episode_master_seed(run_seed, episode)
            clash = seen.get(master)
            assert clash is None, (
                f"worker {worker} episode {episode} reuses the seed of worker "
                f"{clash[0]} episode {clash[1]}"
            )
            seen[master] = (worker, episode)


def test_episode_numbers_do_not_land_on_stream_identities():
    """Episodes are numbered by the caller and would otherwise share the space
    that names the streams, putting episode 5 on RECOMMENDER's seed."""
    run_seed = 42
    identities = {
        substream(run_seed, s) for s in (*STREAMS, DECK_CHOICE, RECOMMENDER, SNAPSHOT_CHOICE)
    }
    for episode in range(1, 51):
        assert episode_master_seed(run_seed, episode) not in identities, (
            f"episode {episode} collides with a stream identity of the same seed"
        )


def test_engine_state_words_are_not_transforms_of_each_other():
    """xoshiro256** advances its state linearly over GF(2).

    Words related by a fixed XOR or a shared multiplier carry that structure
    into its early output, so the four must not be derivable from one another.
    """
    masters = [1, 42, 12345, 999983, 2**40, 7, 88888]
    xors = set()
    diffs = set()
    for master in masters:
        w0, w1, _w2, w3 = state_words(master, ENGINE, 4)
        assert len({w0, w1, _w2, w3}) == 4, f"master {master} repeats a state word"
        xors.add(w0 ^ w3)
        diffs.add((w1 - w0) & 0xFFFFFFFFFFFFFFFF)
    assert len(xors) == len(masters), "w0 ^ w3 is fixed across masters"
    assert len(diffs) == len(masters), "w1 - w0 is fixed across masters"


def test_state_words_are_a_pure_function_and_stream_scoped():
    assert state_words(42, ENGINE, 4) == state_words(42, ENGINE, 4)
    assert state_words(42, ENGINE, 4) != state_words(42, DECK_ORDER, 4)
    assert state_words(42, ENGINE, 4)[0] == substream(42, ENGINE)


def test_a_stream_identity_may_reuse_the_episode_prefix():
    """Episodes are separated from streams by key width, not by a reserved value.

    Their keys are two elements wide, and ``(n, e)`` is distinct from ``(n,)``
    for every ``e``, so the stream identities stay free to grow into ``n``.
    """
    master = 12345
    episodes = {episode_master_seed(master, e) for e in range(500)}
    assert substream(master, _EPISODE_NS) not in episodes, (
        "a stream identity equal to the episode prefix collides with an episode"
    )
