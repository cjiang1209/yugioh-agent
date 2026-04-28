"""Shared CLI helpers used by ``cli.train`` and ``cli.eval``.

Centralizes argument validation and device resolution so both CLIs reject
bad input with identical error strings (pinned by
``tests/cli/test_validate_args.py`` and ``tests/cli/test_eval_cli_validation.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def fatal(msg: str) -> None:
    """Print an error message to stderr and exit with status 2."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def was_provided(name: str) -> bool:
    """Whether a CLI flag was explicitly passed (``--flag value`` or ``--flag=value``)."""
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def validate_opponent_spec(spec: str, flag: str) -> None:
    """Validate an opponent spec string like 'greedy' or 'model:path.pt'."""
    if spec.startswith("model:"):
        path = spec[len("model:"):]
        if not path:
            fatal(f"{flag} model: entries must include a checkpoint path")
        if not Path(path).exists():
            fatal(f"{flag} opponent checkpoint not found: {path}")
    elif spec not in ("greedy", "random"):
        fatal(f"unknown opponent: {spec}")


def validate_deck_paths(paths: list[str], flag: str = "--deck-paths") -> None:
    """Validate that every deck file exists and ends with .ydk."""
    for dp in paths:
        if not Path(dp).exists():
            fatal(f"{flag}: deck file not found: {dp}")
        if not dp.endswith(".ydk"):
            fatal(f"{flag}: deck file must end with .ydk: {dp}")


def resolve_device(spec: str) -> str:
    """Resolve a ``--device`` value to a concrete ``"cpu"`` or ``"cuda"``.

    ``"auto"`` picks cuda when available, else cpu. Concrete strings pass
    through unchanged. The standalone eval CLI must call this before any
    ``torch.device(...)`` / ``torch.load(map_location=...)`` consumer because
    those raise on the literal string ``"auto"``.
    """
    if spec == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec
