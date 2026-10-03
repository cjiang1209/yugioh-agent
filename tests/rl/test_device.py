"""Tests for rl.device.resolve_device."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from rl.device import resolve_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_prefers_cuda_then_mps_then_cpu(cuda, mps, expected):
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert resolve_device("auto") == expected


@pytest.mark.parametrize("spec", ["cpu", "cuda", "mps"])
def test_concrete_spec_passes_through(spec):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device(spec) == spec
