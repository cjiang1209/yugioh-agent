"""Pure Elo rating math. No torch, no shared memory — just the formulas."""
from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B given Elo ratings. Returns 0.5 at parity."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update(
    rating_a: float,
    rating_b: float,
    agent_won: bool,
    k: float = 16.0,
) -> tuple[float, float]:
    """Apply a single-game Elo update. Returns (new_a, new_b).

    Zero-sum: ``new_a + new_b == rating_a + rating_b``.
    """
    expected_a = expected_score(rating_a, rating_b)
    score_a = 1.0 if agent_won else 0.0
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta
