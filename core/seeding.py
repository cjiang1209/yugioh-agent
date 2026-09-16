"""Independent PRNG streams from one master seed.

Drawing two things from one seed ties them together: ``Random(s).randint(...)``
and ``Random(s).shuffle(...)`` read the same first draws, so one result predicts
the other. Each caller takes its own named stream instead.

``spawn_key`` names a child directly, so a stream is a pure function of
``(master, stream)`` -- no parent object to keep alive and no dependence on the
order callers happen to run in.
"""

from __future__ import annotations

from numpy.random import SeedSequence

# Stream identities. These values are part of the reproducibility contract:
# changing one changes every duel derived from it.
SEAT = 0
OPPONENT = 1
DECK_ORDER = 2  # the main deck's opening order
ENGINE = 3  # ygopro-core's internal RNG
DECK_CHOICE = 4  # which deck each side plays
RECOMMENDER = 5
SNAPSHOT_CHOICE = 6  # which self-play snapshot an episode faces

# Episodes are numbered by the caller, so a bare episode number would land on a
# stream identity -- episode 5 on RECOMMENDER's seed. Prefixing widens their
# spawn key to two elements, and (n, e) is distinct from (n,) for every e, so
# this value costs the stream identities above nothing.
_EPISODE_NS = 7

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _words(master: int, spawn_key: tuple[int, ...], count: int) -> list[int]:
    """``count`` 64-bit words from one expansion of a named child sequence.

    ``master`` is taken modulo 2**64, so the negative seeds a CLI flag or a
    request body can carry are usable entropy rather than an error.
    """
    raw = SeedSequence(master & _MASK64, spawn_key=spawn_key).generate_state(
        2 * count, dtype="uint32"
    )
    return [(int(raw[2 * i + 1]) << 32) | int(raw[2 * i]) for i in range(count)]


def substream(master: int, stream: int) -> int:
    """A 64-bit seed for ``stream``, independent of its siblings."""
    return _words(master, (stream,), 1)[0]


def state_words(master: int, stream: int, count: int) -> list[int]:
    """``count`` 64-bit words for ``stream``, each independent of the others.

    For generators whose state is several words wide. xoshiro256** advances its
    state linearly over GF(2), so words related by a fixed XOR or a shared
    multiplier carry structure into its early output; these come from one
    expansion of the stream instead.
    """
    return _words(master, (stream,), count)


def episode_master_seed(run_seed: int, episode: int) -> int:
    """The master seed for one episode, which every stream in it derives from.

    Numbering episodes off a run seed by addition makes a worker's episode N
    reachable as another worker's episode N - offset, so parallel workers replay
    each other's duels once their episode counts pass the offset between their
    run seeds. Naming the episode instead keeps them apart at any count.
    """
    return _words(run_seed, (_EPISODE_NS, episode), 1)[0]
