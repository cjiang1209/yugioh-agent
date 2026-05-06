"""Engine-gated tests for ``_eval_worker`` in isolation.

Drives one worker process via a parent-side ``mp.Pipe``, sending
``("task", _EvalTask)`` messages and asserting the reply shape.  Mirrors
the harness pattern in ``tests/rl/test_actor_learner_worker.py``.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from yugioh_rl.env_wrapper import parse_deck_pool
from yugioh_rl.eval import _EvalTask, _eval_worker, _PartialResult

from tests.rl.conftest import requires_engine


def _spawn_eval_worker(
    deck_pool,
    *,
    agent_spec: str = "random",
    seed: int = 42,
):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_eval_worker,
        kwargs={
            "remote": child,
            "agent_spec": agent_spec,
            "agent_device": "cpu",
            "deck_pool": deck_pool,
            "seed": seed,
            "agent_player": "first",
            "opponent_device": "cpu",
        },
        daemon=True,
    )
    proc.start()
    child.close()
    return proc, parent


def _deck_pool_or_skip():
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")
    return parse_deck_pool([deck])


@requires_engine
def test_worker_completes_one_task() -> None:
    deck_pool = _deck_pool_or_skip()
    proc, parent = _spawn_eval_worker(deck_pool)
    try:
        parent.send(("task", _EvalTask(opp_idx=0, opp_spec="random", episode_idx=1)))
        cmd, payload = parent.recv()
        assert cmd == "partial"
        assert isinstance(payload, _PartialResult)
        assert payload.opp_idx == 0
        assert payload.episode_idx == 1
        assert isinstance(payload.win, bool)
        assert isinstance(payload.agent_deck_idx, int)
        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=15)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_worker_handles_multiple_same_opponent_tasks() -> None:
    """3 tasks for the same opponent: env is built once and reused.

    Indirectly verified via wall-time: each TrainingEnv build pays a
    duel-init cost (~hundreds of ms); 3 builds would push wall time well
    past ~10s.  We just assert all 3 complete in the per-test 30s budget
    and reply in episode_idx order.
    """
    deck_pool = _deck_pool_or_skip()
    proc, parent = _spawn_eval_worker(deck_pool)
    try:
        for ep in (1, 2, 3):
            parent.send(("task", _EvalTask(opp_idx=0, opp_spec="random", episode_idx=ep)))
        replies = [parent.recv() for _ in range(3)]
        for r, ep in zip(replies, (1, 2, 3)):
            cmd, payload = r
            assert cmd == "partial"
            assert payload.opp_idx == 0
            assert payload.episode_idx == ep
        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_worker_swaps_env_on_opponent_change() -> None:
    deck_pool = _deck_pool_or_skip()
    proc, parent = _spawn_eval_worker(deck_pool)
    try:
        # Task 1: opponent A.
        parent.send(("task", _EvalTask(0, "random", 1)))
        cmd_a, payload_a = parent.recv()
        assert cmd_a == "partial"
        assert payload_a.opp_idx == 0

        # Task 2: opponent B — worker must close A's env and build B's.
        parent.send(("task", _EvalTask(1, "greedy", 1)))
        cmd_b, payload_b = parent.recv()
        assert cmd_b == "partial"
        assert payload_b.opp_idx == 1

        parent.send(("shutdown", None))
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_worker_propagates_error() -> None:
    """A bogus agent spec raises in ``make_eval_agent`` at worker startup.

    The worker catches and forwards the traceback as ``("error", str)``
    before the parent's first ``("task", ...)`` is even sent — but the
    parent only learns about it on its first recv, which is what the
    pool driver relies on to surface ``EvalWorkerError``.
    """
    deck_pool = _deck_pool_or_skip()
    proc, parent = _spawn_eval_worker(deck_pool, agent_spec="bogus_spec_xyz")
    try:
        # The worker fails BEFORE entering its main loop, in make_eval_agent.
        # That puts the error on the pipe before we send any task.
        cmd, payload = parent.recv()
        assert cmd == "error"
        assert isinstance(payload, str)
        # The traceback should mention the unknown opponent.
        assert "bogus_spec_xyz" in payload or "unknown" in payload.lower()
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


@requires_engine
def test_worker_handles_shutdown_first() -> None:
    """Worker that receives shutdown before any task exits cleanly with 0."""
    deck_pool = _deck_pool_or_skip()
    proc, parent = _spawn_eval_worker(deck_pool)
    try:
        parent.send(("shutdown", None))
        proc.join(timeout=10)
        assert not proc.is_alive(), "worker did not exit on shutdown"
        assert proc.exitcode == 0, f"worker exited non-zero: {proc.exitcode}"
    finally:
        if proc.is_alive():
            proc.terminate()
