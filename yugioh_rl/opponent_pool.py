"""Snapshot-pool self-play for PPO training.

SharedPoolState owns cross-process shared memory (total_adds counter +
K SharedPolicyWeights slots). OpponentPool is the per-process consumer:
sampling, scripted->snapshot transitions, resume reconstruction.
"""
from __future__ import annotations

import random as stdlib_random
import re
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from yugioh_rl.shared_weights import SharedPolicyWeights
from yugioh_env.opponent import (
    NetworkOpponent,
    Opponent,
    make_opponent,
)


_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


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
      - total_adds: monotonic add counter (one int64 in mp.shared_memory)
      - slots: list of K SharedPolicyWeights instances
    """

    def __init__(
        self,
        total_adds_tensor: torch.Tensor,
        slots: list[SharedPolicyWeights],
    ) -> None:
        self._total_adds = total_adds_tensor
        self.slots = slots

    @classmethod
    def create(cls, pool_size: int, network: nn.Module) -> "SharedPoolState":
        """Trainer-side: allocate shared memory for counter + K weight slots."""
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        total_adds = torch.zeros(1, dtype=torch.int64)
        total_adds.share_memory_()
        slots = [SharedPolicyWeights(network) for _ in range(pool_size)]
        return cls(total_adds, slots)

    @classmethod
    def from_handles(cls, handles: dict[str, Any]) -> "SharedPoolState":
        """Worker-side: attach to existing shared memory."""
        total_adds = handles["total_adds"]
        slots = [SharedPolicyWeights.from_handles(h) for h in handles["slots"]]
        return cls(total_adds, slots)

    def share_handles(self) -> dict[str, Any]:
        """Trainer-side: return picklable handles for worker spawn."""
        return {
            "total_adds": self._total_adds,
            "slots": [s.share_handles() for s in self.slots],
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
    ) -> None:
        self._shared = shared
        self._network_factory = network_factory
        self._temperature = temperature
        self._rng = rng
        self._pool: list[Opponent | None] = [None] * shared.pool_size
        self._local_versions: list[int] = [0] * shared.pool_size

    @classmethod
    def create_trainer(
        cls,
        pool_size: int,
        initial_opponent_spec: str,
        network_factory: Callable[[], nn.Module],
        temperature: float = 1.0,
        rng: stdlib_random.Random | None = None,
    ) -> "OpponentPool":
        """Trainer-side: allocate shared memory, seed slot 0 with initial opponent."""
        # Build a template network to size the SharedPolicyWeights slots.
        template = network_factory()
        shared = SharedPoolState.create(pool_size, template)
        pool = cls(
            shared=shared,
            network_factory=network_factory,
            temperature=temperature,
            rng=rng or stdlib_random.Random(),
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
        rng: stdlib_random.Random | None = None,
    ) -> "OpponentPool":
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
        rng: stdlib_random.Random | None = None,
    ) -> "OpponentPool":
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
            rng=rng,
        )

        aligned = [
            n for n in _find_numbered_checkpoints(checkpoint_dir)
            if n % save_interval == 0
        ]
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
        """
        cur = self._shared.total_adds
        slot = cur % self._shared.pool_size
        self._shared.slots[slot].publish(network)
        self._shared.total_adds = cur + 1
        return slot

    def sample(self) -> Opponent:
        """Refresh stale snapshot slots, then uniformly sample an occupied slot."""
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

        slot = self._rng.randrange(occupied)
        return self._pool[slot]
