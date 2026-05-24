"""Tests for SharedPolicyWeights transport buffer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from yugioh_rl.shared_weights import SharedPolicyWeights


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)


def test_publish_refresh_round_trip() -> None:
    src = _Tiny()
    dst = _Tiny()  # different random init
    shared = SharedPolicyWeights(src)

    shared.publish(src)
    shared.refresh_into(dst)

    for (n1, p1), (n2, p2) in zip(
        sorted(src.state_dict().items()), sorted(dst.state_dict().items())
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"mismatch on {n1}"


def test_seqlock_retries_on_version_mismatch(monkeypatch) -> None:
    """Force a torn version read; verify refresh retries and succeeds."""
    src = _Tiny()
    dst = _Tiny()
    shared = SharedPolicyWeights(src)
    shared.publish(src)

    real_item = torch.Tensor.item
    call_count = {"n": 0}

    def flaky_item(self):
        call_count["n"] += 1
        # On the second call (the v_after read of the first attempt),
        # return a value one higher to simulate a publish racing in.
        if call_count["n"] == 2:
            return real_item(self) + 1
        return real_item(self)

    monkeypatch.setattr(torch.Tensor, "item", flaky_item)

    v = shared.refresh_into(dst)
    assert v == 1
    assert call_count["n"] >= 4, "retry path should have been taken"


def test_version_increments_monotonically() -> None:
    m = _Tiny()
    shared = SharedPolicyWeights(m)
    assert shared.version == 0
    shared.publish(m)
    assert shared.version == 1
    shared.publish(m)
    assert shared.version == 2


def test_nan_publish_rejected() -> None:
    from yugioh_rl.shared_weights import NonFiniteWeightsError

    m = _Tiny()
    shared = SharedPolicyWeights(m)
    with torch.no_grad():
        m.fc.weight.fill_(float("nan"))
    with pytest.raises(NonFiniteWeightsError):
        shared.publish(m)
    assert shared.version == 0  # version did not advance


def _child_refresh(handles, pipe):
    # Runs in a spawned process. Reconstructs SharedPolicyWeights from
    # handles, refreshes a fresh model, and sends the result via pipe.
    import torch.nn as nn

    from yugioh_rl.shared_weights import SharedPolicyWeights

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

    shared = SharedPolicyWeights.from_handles(handles)
    model = _Tiny()
    version = shared.refresh_into(model)
    pipe.send((version, model.fc.weight.detach().clone()))
    # Wait for parent ack before exiting: torch tensors travel via a
    # resource-sharer UNIX socket served by this process, which is torn
    # down when the process exits. Exiting before the parent rebuilds the
    # storage fd produces a FileNotFoundError in rebuild_storage_fd.
    pipe.recv()
    pipe.close()


def test_cross_process_refresh() -> None:
    import multiprocessing as mp

    src = _Tiny()
    shared = SharedPolicyWeights(src)
    shared.publish(src)
    expected_weight = src.fc.weight.detach().clone()

    ctx = mp.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe()
    proc = ctx.Process(target=_child_refresh, args=(shared.share_handles(), child_pipe))
    proc.start()
    child_pipe.close()  # parent uses parent_pipe only

    version, child_weight = parent_pipe.recv()
    parent_pipe.send("ack")  # release the child so it can exit
    proc.join(timeout=30)
    assert proc.exitcode == 0, "child process exited non-zero"

    assert version == 1
    assert torch.equal(child_weight, expected_weight)
