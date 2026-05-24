"""Snapshot-pool self-play for PPO training.

SharedPoolState owns cross-process shared memory (total_adds counter +
K SharedPolicyWeights slots). OpponentPool is the per-process consumer:
sampling, scripted->snapshot transitions, resume reconstruction.
"""

from __future__ import annotations

import random as stdlib_random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args

import torch
import torch.nn as nn

from yugioh_env.opponent import (
    NetworkOpponent,
    Opponent,
    make_opponent,
)
from yugioh_rl.elo import expected_score
from yugioh_rl.elo import update as elo_update
from yugioh_rl.shared_weights import SharedPolicyWeights

_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")

Sampling = Literal["uniform", "pfsp"]
SAMPLING_CHOICES: tuple[str, ...] = get_args(Sampling)
_VALID_SAMPLING = SAMPLING_CHOICES
_PFSP_P = 2.0
_PFSP_EPSILON = 0.2


def _find_numbered_checkpoints(checkpoint_dir: Path) -> list[int]:
    """Return sorted list of update numbers for ``checkpoint_<N>.pt`` files.

    Excludes ``checkpoint_latest.pt`` (always a symlink to a numbered file)
    and any non-matching files.
    """
    if not checkpoint_dir.is_dir():
        return []
    numbers = []
    for entry in checkpoint_dir.iterdir():
        m = _CHECKPOINT_RE.match(entry.name)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)


class SharedPoolState:
    """Cross-process shared state for OpponentPool.

    Owns:
      - total_adds: monotonic add counter (one int64 in shared memory)
      - slots: list of K SharedPolicyWeights instances
      - agent_rating: live Elo rating of the trainer's policy (scalar float32)
      - ratings: per-slot Elo ratings of pool opponents (float32, length K)
      - n_games: per-slot count of games played vs each opponent (int64, length K)
    """

    _DEFAULT_RATING = 1500.0

    def __init__(
        self,
        total_adds_tensor: torch.Tensor,
        slots: list[SharedPolicyWeights],
        agent_rating_tensor: torch.Tensor,
        ratings_tensor: torch.Tensor,
        n_games_tensor: torch.Tensor,
    ) -> None:
        self._total_adds = total_adds_tensor
        self.slots = slots
        self._agent_rating = agent_rating_tensor
        self._ratings = ratings_tensor
        self._n_games = n_games_tensor

    @classmethod
    def create(cls, pool_size: int, network: nn.Module) -> SharedPoolState:
        """Trainer-side: allocate shared memory for counter + K weight slots + Elo state."""
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        total_adds = torch.zeros(1, dtype=torch.int64)
        total_adds.share_memory_()
        slots = [SharedPolicyWeights(network) for _ in range(pool_size)]
        agent_rating = torch.tensor([cls._DEFAULT_RATING], dtype=torch.float32)
        agent_rating.share_memory_()
        ratings = torch.full((pool_size,), cls._DEFAULT_RATING, dtype=torch.float32)
        ratings.share_memory_()
        n_games = torch.zeros(pool_size, dtype=torch.int64)
        n_games.share_memory_()
        return cls(total_adds, slots, agent_rating, ratings, n_games)

    @classmethod
    def from_handles(cls, handles: dict[str, Any]) -> SharedPoolState:
        """Worker-side: attach to existing shared memory."""
        return cls(
            total_adds_tensor=handles["total_adds"],
            slots=[SharedPolicyWeights.from_handles(h) for h in handles["slots"]],
            agent_rating_tensor=handles["agent_rating"],
            ratings_tensor=handles["ratings"],
            n_games_tensor=handles["n_games"],
        )

    def share_handles(self) -> dict[str, Any]:
        """Trainer-side: return picklable handles for worker spawn."""
        return {
            "total_adds": self._total_adds,
            "slots": [s.share_handles() for s in self.slots],
            "agent_rating": self._agent_rating,
            "ratings": self._ratings,
            "n_games": self._n_games,
        }

    @property
    def total_adds(self) -> int:
        return int(self._total_adds[0].item())

    @total_adds.setter
    def total_adds(self, value: int) -> None:
        self._total_adds[0] = value

    @property
    def pool_size(self) -> int:
        return len(self.slots)

    @property
    def agent_rating(self) -> float:
        return float(self._agent_rating[0].item())

    @agent_rating.setter
    def agent_rating(self, value: float) -> None:
        self._agent_rating[0] = float(value)

    def get_rating(self, slot: int) -> float:
        return float(self._ratings[slot].item())

    def set_rating(self, slot: int, value: float) -> None:
        self._ratings[slot] = float(value)

    def get_n_games(self, slot: int) -> int:
        return int(self._n_games[slot].item())

    def increment_n_games(self, slot: int) -> None:
        self._n_games[slot] += 1

    def n_games_zero(self, slot: int) -> None:
        self._n_games[slot] = 0


class OpponentPool:
    """Ring-buffer opponent pool for self-play training.

    Trainer uses create_trainer() (allocates shared memory and seeds
    slot 0 with the initial opponent). Each worker uses attach_worker()
    with the handles dict from share_handles(). All cross-process
    synchronization is internal — callers don't deal with shared memory.
    """

    def __init__(
        self,
        shared: SharedPoolState,
        network_factory: Callable[[], nn.Module],
        temperature: float,
        rng: stdlib_random.Random,
        sampling: Sampling = "uniform",
    ) -> None:
        if sampling not in _VALID_SAMPLING:
            raise ValueError(f"sampling must be one of {_VALID_SAMPLING}, got {sampling!r}")
        self._shared = shared
        self._network_factory = network_factory
        self._temperature = temperature
        self._rng = rng
        self._sampling = sampling
        self._pool: list[Opponent | None] = [None] * shared.pool_size
        self._local_versions: list[int] = [0] * shared.pool_size

    @classmethod
    def create_trainer(
        cls,
        pool_size: int,
        initial_opponent_spec: str,
        network_factory: Callable[[], nn.Module],
        temperature: float = 1.0,
        sampling: Sampling = "uniform",
        rng: stdlib_random.Random | None = None,
    ) -> OpponentPool:
        """Trainer-side: allocate shared memory, seed slot 0 with initial opponent."""
        # Build a template network to size the SharedPolicyWeights slots.
        template = network_factory()
        shared = SharedPoolState.create(pool_size, template)
        pool = cls(
            shared=shared,
            network_factory=network_factory,
            temperature=temperature,
            rng=rng or stdlib_random.Random(),
            sampling=sampling,
        )
        pool._pool[0] = make_opponent(initial_opponent_spec)
        shared.total_adds = 1
        return pool

    @classmethod
    def attach_worker(
        cls,
        handles: dict[str, Any],
        initial_opponent_spec: str,
        network_factory: Callable[[], nn.Module],
        temperature: float = 1.0,
        sampling: Sampling = "uniform",
        rng: stdlib_random.Random | None = None,
    ) -> OpponentPool:
        """Worker-side: attach to existing SharedPoolState via handles dict.

        If slot 0 still holds the initial scripted opponent (no snapshot
        has overwritten it yet — detected by slot.version == 0), reconstruct
        it locally from the same spec. Snapshot slots are lazy-filled in
        sample() via the refresh loop.
        """
        shared = SharedPoolState.from_handles(handles)
        pool = cls(
            shared=shared,
            network_factory=network_factory,
            temperature=temperature,
            rng=rng or stdlib_random.Random(),
            sampling=sampling,
        )
        # Slot 0 still scripted iff total_adds >= 1 and its version is still 0.
        if shared.total_adds >= 1 and shared.slots[0].version == 0:
            pool._pool[0] = make_opponent(initial_opponent_spec)
        return pool

    @classmethod
    def from_resume(
        cls,
        pool_size: int,
        initial_opponent_spec: str,
        network_factory: Callable[[], nn.Module],
        save_interval: int,
        checkpoint_dir: Path,
        temperature: float = 1.0,
        sampling: Sampling = "uniform",
        rng: stdlib_random.Random | None = None,
    ) -> OpponentPool:
        """Trainer-side construction from disk checkpoints.

        Replays add_snapshot in chronological order to reconstruct the pool
        a continuous run would have at this point. Only interval-aligned
        checkpoints participate; off-interval crash saves are skipped.
        """
        pool = cls.create_trainer(
            pool_size=pool_size,
            initial_opponent_spec=initial_opponent_spec,
            network_factory=network_factory,
            temperature=temperature,
            sampling=sampling,
            rng=rng,
        )

        aligned = [n for n in _find_numbered_checkpoints(checkpoint_dir) if n % save_interval == 0]
        recent = aligned[-pool_size:]

        if len(recent) >= pool_size:
            # Enough snapshots to fill the pool; start fresh without the
            # initial seed.
            pool._shared.total_adds = 0
            pool._pool[0] = None

        for n in recent:
            net = network_factory()
            ckpt = torch.load(
                checkpoint_dir / f"checkpoint_{n}.pt",
                map_location="cpu",
                weights_only=False,
            )
            net.load_state_dict(ckpt["model_state_dict"])
            pool.add_snapshot(net)

        return pool

    def share_handles(self) -> dict[str, Any]:
        return self._shared.share_handles()

    def occupied_count(self) -> int:
        return min(self._shared.total_adds, self._shared.pool_size)

    def add_snapshot(self, network: nn.Module) -> int:
        """Publish ``network``'s weights into the next ring slot. Returns slot id.

        Trainer-only. The slot's version is bumped (via SharedPolicyWeights.publish)
        before total_adds is incremented, so a worker reading total_adds = N is
        guaranteed slot (N-1) % pool_size was fully published.

        Inherits the current agent_rating into the new slot (standard self-play
        convention: a snapshot of the trainer is, at publish time, as strong as
        the trainer). Resets the slot's n_games to 0.
        """
        cur = self._shared.total_adds
        slot = cur % self._shared.pool_size
        self._shared.slots[slot].publish(network)
        self._shared.set_rating(slot, self._shared.agent_rating)
        self._shared.n_games_zero(slot)
        self._shared.total_adds = cur + 1
        return slot

    def sample(self) -> tuple[int, Opponent]:
        """Refresh stale snapshot slots, then uniformly sample an occupied slot.

        Returns (slot_id, opponent) — callers use slot_id to report match
        outcomes back via :meth:`report_result` for Elo updates.
        """
        occupied = self.occupied_count()
        if occupied == 0:
            raise RuntimeError("OpponentPool.sample() called on empty pool")

        for i in range(occupied):
            slot_version = self._shared.slots[i].version
            if slot_version <= self._local_versions[i]:
                continue

            existing = self._pool[i]
            if isinstance(existing, NetworkOpponent):
                # In-place weight refresh on the cached network.
                self._shared.slots[i].refresh_into(existing.network)
            else:
                # First snapshot in this slot — replace whatever was here.
                net = self._network_factory()
                self._shared.slots[i].refresh_into(net)
                self._pool[i] = NetworkOpponent(
                    net,
                    stochastic=True,
                    temperature=self._temperature,
                )
            self._local_versions[i] = slot_version

        slot = self._pick_slot(occupied)
        return slot, self._pool[slot]

    def _pick_slot(self, occupied: int) -> int:
        """Strategy dispatcher for slot selection. ``occupied`` is the
        guaranteed-occupied prefix length (slots 0..occupied-1)."""
        if self._sampling == "uniform":
            return self._rng.randrange(occupied)
        if self._sampling == "pfsp":
            return self._pick_slot_pfsp(occupied)
        raise RuntimeError(f"unhandled sampling strategy: {self._sampling!r}")

    def _pick_slot_pfsp(self, occupied: int) -> int:
        """PFSP: weight each occupied slot by ``(1 - P(agent beats it))^p``,
        mix in ``epsilon`` uniform exploration, fall back to uniform if all
        weights collapse (agent dominates every slot).

        Reads ``agent_rating`` and ``ratings[]`` from shared memory non-
        atomically — see CLAUDE.md note on the K-bounded drift trade-off.
        """
        if self._rng.random() < _PFSP_EPSILON:
            return self._rng.randrange(occupied)

        agent = self._shared.agent_rating
        weights = [
            (1.0 - expected_score(agent, self._shared.get_rating(i))) ** _PFSP_P
            for i in range(occupied)
        ]
        if sum(weights) <= 1e-9:
            return self._rng.randrange(occupied)
        return self._rng.choices(range(occupied), weights=weights, k=1)[0]

    def report_result(self, slot: int, agent_won: bool, k: float = 16.0) -> None:
        """Apply an Elo update for one episode vs ``slot``.

        Not atomic across workers: two simultaneous updates can race on
        ``agent_rating``. Acceptable for a logging metric; the typical
        drift is bounded by K per dropped update and self-corrects on the
        next match.
        """
        agent = self._shared.agent_rating
        opp = self._shared.get_rating(slot)
        new_agent, new_opp = elo_update(agent, opp, agent_won=agent_won, k=k)
        self._shared.agent_rating = new_agent
        self._shared.set_rating(slot, new_opp)
        self._shared.increment_n_games(slot)

    def elo_summary(self) -> dict[str, float | int]:
        """Snapshot of current ratings for logging. Only counts occupied slots.

        Returns NaN for pool_mean/min/max when no slots are occupied so
        TensorBoard skips the points instead of plotting a fake 0.0.
        """
        occupied = self.occupied_count()
        if occupied == 0:
            nan = float("nan")
            return {
                "agent": self._shared.agent_rating,
                "pool_mean": nan,
                "pool_min": nan,
                "pool_max": nan,
                "occupied": 0,
            }
        ratings = [self._shared.get_rating(i) for i in range(occupied)]
        return {
            "agent": self._shared.agent_rating,
            "pool_mean": sum(ratings) / len(ratings),
            "pool_min": min(ratings),
            "pool_max": max(ratings),
            "occupied": occupied,
        }
