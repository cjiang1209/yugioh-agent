"""Parent-side error-taxonomy tests for ``_run_eval_pool``.

Drives the pool with module-level fake workers (passed via ``worker_fn``)
that simulate specific failure modes — crashed-before-reply, silent past
timeout, mid-task crash, and explicit Python exception.  No engine
dependency: the fakes never construct a ``TrainingEnv``.

Module-level definitions are required so ``mp.get_context("spawn")`` can
pickle them across the process boundary.
"""

from __future__ import annotations

import os
import time

import pytest

from yugioh_rl.actor_learner import WorkerDiedError, WorkerTimeoutError
from yugioh_rl.eval import EvalWorkerError, _PartialResult, _run_eval_pool

# ---------------------------------------------------------------------------
# Fake worker entrypoints — all match the _eval_worker signature.
# ---------------------------------------------------------------------------


def _fake_worker_crashes_before_reply(remote, **_kwargs) -> None:
    """Exit immediately without ever touching the pipe.  Parent's first
    recv on this worker should surface as WorkerDiedError."""
    os._exit(1)


def _fake_worker_silent(remote, **_kwargs) -> None:
    """Receive the task but never reply.  Parent's wait() times out;
    proc.is_alive() is True, so the parent must raise WorkerTimeoutError."""
    cmd, _ = remote.recv()
    if cmd == "shutdown":
        return
    time.sleep(60.0)


def _fake_worker_replies_then_crashes(remote, **_kwargs) -> None:
    """Reply once, then exit before the parent's replenish-send arrives.
    Tests the second send site's WorkerDiedError wrapping."""
    cmd, payload = remote.recv()
    assert cmd == "task"
    remote.send(("partial", _PartialResult(payload.opp_idx, payload.episode_idx, True, 0)))
    # Crash before reading the next task. Sleep so the parent has time
    # to receive the partial and call _send_task before we exit (a
    # too-short window can race on a loaded CI box).
    time.sleep(0.3)
    os._exit(1)


def _fake_worker_sends_error(remote, **_kwargs) -> None:
    """Mimic a Python exception inside _eval_worker — sends an error tuple."""
    cmd, _ = remote.recv()
    if cmd == "shutdown":
        return
    remote.send(("error", "fake traceback marker"))


def _fake_worker_dies_on_first_send(remote, **_kwargs) -> None:
    """Exit before reading any pipe message.  Parent's initial dispatch
    send should hit BrokenPipeError → WorkerDiedError."""
    # Sleep briefly so the parent's send is in flight when we exit;
    # without the sleep, the parent might race and finish the send before
    # the kernel marks the pipe as closed.
    time.sleep(0.05)
    os._exit(1)


# ---------------------------------------------------------------------------
# Common pool kwargs — fakes ignore most of these but the call requires them.
# ---------------------------------------------------------------------------


_POOL_KWARGS = dict(
    agent_spec="random",
    agent_device="cpu",
    deck_pool=[{"main": [123] * 40, "extra": []}],
    seed=42,
    agent_player="first",
    opponent_device=None,
)


# ---------------------------------------------------------------------------
# Signature contract — guards against drift hidden by **_kwargs in the fakes.
# ---------------------------------------------------------------------------


def test_eval_worker_signature_stable():
    """The fake worker entrypoints in this file ignore unknown kwargs via
    ``**_kwargs``; that hides signature drift between the production
    worker and the kwargs the pool sends.  Pin the exact set of kwargs
    the pool dispatches so renaming/adding/removing a real worker arg
    fails this test loudly instead of silently absorbing into the fakes.
    """
    import inspect

    from yugioh_rl.eval import _eval_worker

    params = inspect.signature(_eval_worker).parameters
    expected = {
        "remote",
        "agent_spec",
        "agent_device",
        "deck_pool",
        "seed",
        "agent_player",
        "opponent_device",
        "deck_allocation",
        "mirror_decks",
    }
    assert set(params) == expected, (
        f"_eval_worker signature drifted: got {set(params)}, expected {expected}. "
        "Update the fake workers in this file (and _POOL_KWARGS) to match."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pool_raises_worker_died_on_recv_when_child_crashed():
    with pytest.raises(WorkerDiedError, match="died"):
        _run_eval_pool(
            opponent_specs=["a"],
            num_episodes=1,
            num_workers=1,
            worker_timeout_s=5.0,
            worker_fn=_fake_worker_crashes_before_reply,
            **_POOL_KWARGS,
        )


def test_pool_raises_worker_died_on_initial_dispatch_when_child_dead():
    with pytest.raises(WorkerDiedError):
        _run_eval_pool(
            opponent_specs=["a"],
            num_episodes=1,
            num_workers=1,
            worker_timeout_s=5.0,
            worker_fn=_fake_worker_dies_on_first_send,
            **_POOL_KWARGS,
        )


def test_pool_raises_worker_died_on_replenish_when_child_dies_mid_run():
    """1 worker, 2 tasks: worker handles task 1 then exits; the replenish
    send for task 2 must convert BrokenPipeError into WorkerDiedError."""
    with pytest.raises(WorkerDiedError):
        _run_eval_pool(
            opponent_specs=["a"],
            num_episodes=2,
            num_workers=1,
            worker_timeout_s=5.0,
            worker_fn=_fake_worker_replies_then_crashes,
            **_POOL_KWARGS,
        )


def test_pool_raises_worker_timeout_on_silent_child():
    with pytest.raises(WorkerTimeoutError, match="silent"):
        _run_eval_pool(
            opponent_specs=["a"],
            num_episodes=1,
            num_workers=1,
            worker_timeout_s=0.5,
            worker_fn=_fake_worker_silent,
            **_POOL_KWARGS,
        )


def test_pool_raises_eval_worker_error_on_child_exception():
    with pytest.raises(EvalWorkerError) as exc_info:
        _run_eval_pool(
            opponent_specs=["a"],
            num_episodes=1,
            num_workers=1,
            worker_timeout_s=5.0,
            worker_fn=_fake_worker_sends_error,
            **_POOL_KWARGS,
        )
    # EvalWorkerError carries (opp_spec, traceback_str) per the plan.
    assert exc_info.value.args[0] == "a"
    assert "fake traceback marker" in exc_info.value.args[1]


def test_pool_cleans_up_remaining_workers_on_failure():
    """When one worker fails, all spawned workers must be joined/terminated.

    Verified by capturing the workers via the spawn-context: after the
    raise, all child processes attached to this Pool must be is_alive=False.
    Approach: run a 4-worker pool against the silent fake; after the
    timeout raise, give a short grace period for finally cleanup, then
    assert no zombies remain in our process tree.
    """
    import multiprocessing as mp

    # Snapshot active children before the call.
    pre = set(p.pid for p in mp.active_children())
    with pytest.raises(WorkerTimeoutError):
        _run_eval_pool(
            opponent_specs=["a", "b", "c", "d"],
            num_episodes=1,
            num_workers=4,
            worker_timeout_s=0.5,
            worker_fn=_fake_worker_silent,
            **_POOL_KWARGS,
        )

    # The finally-block in _run_eval_pool joins/terminates each worker.
    # Allow a brief moment for the OS to reap exited children.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        post = set(p.pid for p in mp.active_children()) - pre
        if not post:
            break
        time.sleep(0.05)
    leaked = set(p.pid for p in mp.active_children()) - pre
    assert not leaked, f"_run_eval_pool leaked workers: {leaked}"
