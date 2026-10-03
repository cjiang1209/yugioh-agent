"""Torch device selection."""

from __future__ import annotations

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
"""Accepted device specs."""


def resolve_device(spec: str) -> str:
    """Resolve a ``--device`` value to a concrete ``"cpu"``, ``"cuda"``, or ``"mps"``.

    ``"auto"`` picks cuda when available, else mps when available, else cpu.
    Concrete strings pass through unchanged.
    """
    if spec == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec
