"""Shared-memory transport for agent policy weights.

The trainer writes its master_policy state_dict into shared CPU tensors via
``publish``; workers copy from those shared tensors into their local policy
via ``refresh_into``. A monotonic version counter (also a shared tensor) lets
workers detect when new weights are available.

Note: on weakly-ordered architectures (Apple Silicon ARM), a worker may
briefly observe the new version with not-yet-visible tensor writes; the
seqlock retry handles this by re-reading the version after the copy.

Concurrency contract is a seqlock-style retry, not a lock: the trainer is
the single writer (publishes are rare, ~once per rollout cycle), and workers
detect publish-during-read by re-checking the version counter after copying
all tensors. This avoids POSIX semaphores (``mp.Lock`` / ``mp.Value``), which
are blocked under hardened-runtime macOS deployments. Retries are bounded
and operationally rare given the publish rate.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class NonFiniteWeightsError(RuntimeError):
    """Raised when ``publish`` is called on a model containing NaN/Inf."""


class StaleReadError(RuntimeError):
    """Raised when ``refresh_into`` cannot get a consistent read in
    ``_MAX_REFRESH_RETRIES`` attempts. Operationally unreachable given the
    expected publish rate; surfaced rather than silently looping forever."""


_MAX_REFRESH_RETRIES = 16


class SharedPolicyWeights:
    """Cross-process transport for a model state_dict.

    Allocate at trainer startup (the trainer's process must be the one that
    creates the shared tensors).  Pass ``share_handles()`` through ``spawn``
    kwargs to workers; workers reconstruct via ``from_handles``.
    """

    def __init__(self, model: nn.Module) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        for name, param in model.state_dict().items():
            t = torch.empty_like(param, device="cpu").detach()
            t.share_memory_()
            self._tensors[name] = t
        self._version = torch.zeros(1, dtype=torch.int64)
        self._version.share_memory_()

    def _read_version(self) -> int:
        return int(self._version[0].item())

    @property
    def version(self) -> int:
        return self._read_version()

    def publish(self, model: nn.Module) -> int:
        """Trainer-side: copy ``model.state_dict()`` into shared tensors and
        bump the version. Raises ``NonFiniteWeightsError`` if any param is
        NaN/Inf (a poisoned publish would corrupt every worker).

        Single-writer assumption: only the trainer process publishes.
        """
        sd = model.state_dict()
        for name, src in sd.items():
            if src.dtype.is_floating_point and not torch.isfinite(src).all():
                raise NonFiniteWeightsError(
                    f"parameter {name!r} contains NaN/Inf - aborting publish"
                )
        for name, dst in self._tensors.items():
            src = sd[name].detach().to("cpu", copy=False)
            dst.copy_(src)
        # Bump only after all writes — workers seeing the new version are
        # guaranteed complete tensors (modulo CPU memory ordering, which
        # the seqlock retry on the worker side covers).
        self._version[0] += 1
        return self._read_version()

    def refresh_into(self, model: nn.Module) -> int:
        """Worker-side: copy shared tensors into ``model`` and return the
        version that was read.

        Uses a seqlock-style retry: read version, copy all tensors, re-read
        version. If the version moved mid-read, retry (a publish raced).
        Bounded by ``_MAX_REFRESH_RETRIES``; raises ``StaleReadError`` on
        exhaustion.
        """
        sd = model.state_dict()
        for _ in range(_MAX_REFRESH_RETRIES):
            v_before = self._read_version()
            for name, src in self._tensors.items():
                sd[name].copy_(src)
            v_after = self._read_version()
            if v_before == v_after:
                return v_after
        raise StaleReadError(
            f"could not get a consistent read after {_MAX_REFRESH_RETRIES} retries"
        )

    def share_handles(self) -> dict[str, Any]:
        """Return a picklable bundle of state for ``from_handles``."""
        return {"tensors": self._tensors, "version": self._version}

    @classmethod
    def from_handles(cls, handles: dict[str, Any]) -> "SharedPolicyWeights":
        instance = cls.__new__(cls)
        instance._tensors = handles["tensors"]
        instance._version = handles["version"]
        return instance
