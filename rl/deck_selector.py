"""Pure deck-index selection for training and eval (engine/torch-free).

Deterministic function of (seed, episode_idx, allocation, mirror). Index-level
only: never resolves player seat or maps decks to engine players — consumers do
that. Reproducible, not byte-identical to any prior sequence.
"""

from __future__ import annotations

import random


class DeckSelector:
    def __init__(
        self, pool_size: int, seed: int, *, allocation: str = "random", mirror: bool = False
    ) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if allocation not in ("random", "balanced"):
            raise ValueError(f"unknown allocation: {allocation!r}")
        self._n = pool_size
        self._seed = seed
        self._allocation = allocation
        self._mirror = mirror

    def select(self, episode_idx: int) -> tuple[int, int]:
        """Return (agent_deck_idx, opp_deck_idx) for a 1-indexed episode."""
        rng = random.Random(self._seed + episode_idx)
        if self._allocation == "balanced":
            agent = (episode_idx - 1) % self._n
        else:  # "random"
            agent = rng.randrange(self._n)
        if self._mirror:
            return agent, agent
        opp = rng.randrange(self._n)
        return agent, opp
