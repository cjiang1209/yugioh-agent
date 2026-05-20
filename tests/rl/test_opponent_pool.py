"""Tests for OpponentPool and SharedPoolState (yugioh_rl.opponent_pool)."""
from __future__ import annotations

import multiprocessing as mp
import random as stdlib_random
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from yugioh_rl.opponent_pool import OpponentPool, SharedPoolState
from yugioh_env.opponent import GreedyOpponent, NetworkOpponent, RandomOpponent


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def init_hx(self, batch_size: int, device) -> None:
        return None


# ---------------------------------------------------------------------------
# Task 1: SharedPoolState skeleton + total_adds counter
# ---------------------------------------------------------------------------

def test_shared_pool_state_total_adds_starts_at_zero() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    assert state.total_adds == 0


def test_shared_pool_state_total_adds_round_trips() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.total_adds = 5
    assert state.total_adds == 5


# ---------------------------------------------------------------------------
# Task 2: SharedPoolState cross-process handoff
# ---------------------------------------------------------------------------

def _child_read_total_adds(handles, send_pipe):
    """Runs in spawned process. Reads total_adds via from_handles."""
    from yugioh_rl.opponent_pool import SharedPoolState
    state = SharedPoolState.from_handles(handles)
    send_pipe.send(state.total_adds)
    send_pipe.close()


def test_shared_pool_state_cross_process_total_adds() -> None:
    state = SharedPoolState.create(pool_size=3, network=_Tiny())
    state.total_adds = 7

    ctx = mp.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe()
    proc = ctx.Process(
        target=_child_read_total_adds,
        args=(state.share_handles(), child_pipe),
    )
    proc.start()
    child_pipe.close()
    proc.join(timeout=30)
    assert proc.exitcode == 0

    assert parent_pipe.recv() == 7


# ---------------------------------------------------------------------------
# Task 3: OpponentPool skeleton + initial-opponent seeding
# ---------------------------------------------------------------------------

def test_create_trainer_seeds_slot_zero_with_greedy() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    assert pool.occupied_count() == 1
    assert isinstance(pool._pool[0], GreedyOpponent)
    assert pool._shared.total_adds == 1


def test_create_trainer_seeds_slot_zero_with_random() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="random",
        network_factory=_Tiny,
    )
    assert isinstance(pool._pool[0], RandomOpponent)


def test_occupied_count_clamps_at_pool_size() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=2,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    pool._shared.total_adds = 100
    assert pool.occupied_count() == 2


# ---------------------------------------------------------------------------
# Task 4: OpponentPool.add_snapshot — trainer-side ring-buffer write
# ---------------------------------------------------------------------------

def test_add_snapshot_returns_next_slot_id() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    # After create_trainer, total_adds = 1, so next slot = 1.
    slot = pool.add_snapshot(_Tiny())
    assert slot == 1


def test_add_snapshot_advances_total_adds() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    pool.add_snapshot(_Tiny())
    assert pool._shared.total_adds == 2
    pool.add_snapshot(_Tiny())
    assert pool._shared.total_adds == 3


def test_add_snapshot_wraps_at_pool_size() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    # Initial seed used slot 0 (total_adds=1). Next three publishes go to
    # slots 1, 2, 0 (evicting initial opponent).
    assert pool.add_snapshot(_Tiny()) == 1
    assert pool.add_snapshot(_Tiny()) == 2
    assert pool.add_snapshot(_Tiny()) == 0  # ring wrapped


def test_add_snapshot_bumps_slot_version() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    assert pool._shared.slots[1].version == 0
    pool.add_snapshot(_Tiny())
    assert pool._shared.slots[1].version == 1


# ---------------------------------------------------------------------------
# Task 6: OpponentPool.sample — worker-side refresh + uniform sample
# ---------------------------------------------------------------------------

def test_sample_returns_initial_opponent_when_no_snapshots() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    _, sampled = pool.sample()
    assert isinstance(sampled, GreedyOpponent)


def test_sample_refreshes_after_snapshot_publish() -> None:
    """After add_snapshot, the next sample() that lands on the new slot must
    observe a NetworkOpponent reconstructed from shared memory.

    We don't rely on RNG to pick the snapshot slot — instead, we add a
    snapshot and force the refresh by calling sample() once (which walks all
    occupied slots and refreshes any with version > local_versions[i]).
    After sample(), slot 1 must hold a NetworkOpponent regardless of which
    slot was actually returned by the rng.
    """
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    pool.add_snapshot(_Tiny())
    # Slot 1 has version=1 but the trainer-side _pool[1] is still None.
    assert pool._pool[1] is None
    # Simulate a worker: attach and sample once. The refresh loop must
    # reconstruct slot 1 as a NetworkOpponent.
    worker = OpponentPool.attach_worker(
        handles=pool.share_handles(),
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    worker.sample()  # forces refresh of all occupied slots
    assert isinstance(worker._pool[1], NetworkOpponent)


def test_sample_uniform_distribution_over_occupied() -> None:
    """With pool_size=3 and 3 slots filled, ~33% of samples should hit each slot."""
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        rng=stdlib_random.Random(0),
    )
    pool.add_snapshot(_Tiny())  # slot 1
    pool.add_snapshot(_Tiny())  # slot 2

    # Track sampled slot indices.
    counts = [0, 0, 0]
    for _ in range(3000):
        slot, _ = pool.sample()
        counts[slot] += 1
    # Expect ~1000 each; allow generous tolerance.
    for c in counts:
        assert 800 < c < 1200, f"non-uniform sampling: {counts}"


def test_attach_worker_seeds_slot_zero_from_spec() -> None:
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    worker_pool = OpponentPool.attach_worker(
        handles=pool.share_handles(),
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    assert isinstance(worker_pool._pool[0], GreedyOpponent)
    assert worker_pool.occupied_count() == 1


# ---------------------------------------------------------------------------
# Task 7: Eviction transition — scripted -> snapshot in slot 0
# ---------------------------------------------------------------------------

def test_initial_opponent_evicted_when_ring_wraps() -> None:
    """After K snapshots, slot 0 should hold a NetworkOpponent, not greedy."""
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    # Start: slot 0 = greedy.  total_adds = 1 -> next slot = 1.
    pool.add_snapshot(_Tiny())  # slot 1
    pool.add_snapshot(_Tiny())  # slot 2
    assert isinstance(pool._pool[0], GreedyOpponent)  # still greedy
    pool.add_snapshot(_Tiny())  # slot 0 (ring wrapped; greedy evicted)

    # Trainer-side _pool[0] is still the GreedyOpponent until sample() runs;
    # this is fine because the trainer never calls sample(). But the slot's
    # version is now > 0, so a worker's sample() will rebuild it.
    assert pool._shared.slots[0].version == 1

    # Simulate worker attach + sample to observe the transition.
    worker = OpponentPool.attach_worker(
        handles=pool.share_handles(),
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    # Worker sees total_adds >= 1 but slot 0's version > 0, so it should
    # NOT seed greedy at attach time.
    assert worker._pool[0] is None
    # After sample(), slot 0 must be a NetworkOpponent (the wrapped snapshot).
    worker._rng = stdlib_random.Random(0)
    for _ in range(20):
        worker.sample()  # forces refresh on slots 0, 1, 2
    assert isinstance(worker._pool[0], NetworkOpponent)


# ---------------------------------------------------------------------------
# Task 8: OpponentPool.from_resume — rebuild from disk checkpoints
# ---------------------------------------------------------------------------

def _write_fake_ckpt(path: Path, network: nn.Module) -> None:
    """Write a checkpoint with just the keys from_resume needs."""
    torch.save({"model_state_dict": network.state_dict()}, path)


def test_from_resume_partial_pool_retains_initial(tmp_path: Path) -> None:
    """Fewer than pool_size aligned checkpoints: initial opponent stays."""
    # Write checkpoints at updates 100, 200 (aligned to interval 100).
    net = _Tiny()
    _write_fake_ckpt(tmp_path / "checkpoint_100.pt", net)
    _write_fake_ckpt(tmp_path / "checkpoint_200.pt", net)

    pool = OpponentPool.from_resume(
        pool_size=5,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        save_interval=100,
        checkpoint_dir=tmp_path,
    )
    # Expected: slot 0 = greedy (seed), slots 1, 2 = snapshots from 100, 200.
    assert isinstance(pool._pool[0], GreedyOpponent)
    assert pool._shared.slots[0].version == 0
    assert pool._shared.slots[1].version == 1
    assert pool._shared.slots[2].version == 1
    assert pool._shared.total_adds == 3
    assert pool.occupied_count() == 3


def test_from_resume_full_pool_evicts_initial(tmp_path: Path) -> None:
    """K or more aligned checkpoints: initial opponent is discarded."""
    net = _Tiny()
    for n in [100, 200, 300, 400, 500]:
        _write_fake_ckpt(tmp_path / f"checkpoint_{n}.pt", net)

    pool = OpponentPool.from_resume(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        save_interval=100,
        checkpoint_dir=tmp_path,
    )
    # Pool size 3, 5 aligned checkpoints -> keep last 3 (300, 400, 500).
    # Ring replay: total_adds=0, add(300)->slot 0, add(400)->slot 1,
    # add(500)->slot 2. No initial seed.
    assert pool._shared.total_adds == 3
    assert pool._shared.slots[0].version == 1
    assert pool._shared.slots[1].version == 1
    assert pool._shared.slots[2].version == 1
    # No GreedyOpponent in _pool — slot 0 was overwritten.
    assert not any(isinstance(p, GreedyOpponent) for p in pool._pool if p is not None)


def test_from_resume_skips_off_interval_checkpoints(tmp_path: Path) -> None:
    """Off-interval crash saves (e.g. checkpoint_610.pt) are filtered out."""
    net = _Tiny()
    _write_fake_ckpt(tmp_path / "checkpoint_100.pt", net)
    _write_fake_ckpt(tmp_path / "checkpoint_200.pt", net)
    _write_fake_ckpt(tmp_path / "checkpoint_610.pt", net)  # off-interval; skip

    pool = OpponentPool.from_resume(
        pool_size=5,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        save_interval=100,
        checkpoint_dir=tmp_path,
    )
    # Only 100 and 200 should be added -> slots 1 and 2 populated.
    assert pool._shared.total_adds == 3
    assert pool._shared.slots[1].version == 1
    assert pool._shared.slots[2].version == 1
    assert pool._shared.slots[3].version == 0  # no third snapshot


def test_from_resume_ignores_latest_symlink(tmp_path: Path) -> None:
    """checkpoint_latest.pt is a symlink; filter logic must not double-count it."""
    net = _Tiny()
    _write_fake_ckpt(tmp_path / "checkpoint_100.pt", net)
    (tmp_path / "checkpoint_latest.pt").symlink_to("checkpoint_100.pt")

    pool = OpponentPool.from_resume(
        pool_size=5,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        save_interval=100,
        checkpoint_dir=tmp_path,
    )
    # Only one checkpoint (100); pool has initial + 1 snapshot.
    assert pool._shared.total_adds == 2


def test_from_resume_with_missing_dir(tmp_path: Path) -> None:
    """from_resume with a non-existent checkpoint_dir returns the seeded-only pool."""
    missing = tmp_path / "does_not_exist"
    pool = OpponentPool.from_resume(
        pool_size=5,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
        save_interval=100,
        checkpoint_dir=missing,
    )
    # Only the initial seed is present.
    assert pool._shared.total_adds == 1
    assert isinstance(pool._pool[0], GreedyOpponent)


def test_shared_pool_state_create_rejects_zero_pool_size() -> None:
    with pytest.raises(ValueError, match="pool_size must be >= 1"):
        SharedPoolState.create(pool_size=0, network=_Tiny())


def test_sample_raises_on_empty_pool() -> None:
    """sample() must raise when no slots are occupied (degenerate, but possible
    if total_adds is manually zeroed)."""
    pool = OpponentPool.create_trainer(
        pool_size=3,
        initial_opponent_spec="greedy",
        network_factory=_Tiny,
    )
    # Manually zero the counter to force the empty-pool path.
    pool._shared.total_adds = 0
    with pytest.raises(RuntimeError, match="empty pool"):
        pool.sample()
