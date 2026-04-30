"""Entry file schema, I/O, and entry_id derivation.

An entry is one row on the leaderboard — one checkpoint's panel results
and any pairwise records involving it. Files live at
``leaderboard/entries/<entry_id>.json``. Atomic write (tmp + os.replace)
prevents partial-write corruption.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """UTC ISO-8601 timestamp used by ``added_at`` and ``evaluated_at``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path | str, content: str) -> None:
    """Write ``content`` to ``path`` via tmp + ``os.replace`` for crash safety."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


_CHECKPOINT_RE = re.compile(r"^checkpoint_(?P<suffix>\d+|latest)\.pt$")


@dataclass
class PanelMatchResult:
    opponent_label: str
    episodes: int
    wins: int
    win_rate: float
    per_deck: dict[str, dict[str, float | int]]
    seed: int
    evaluated_at: str


@dataclass
class PairwiseMatchResult:
    vs_entry_id: str
    vs_checkpoint_hash: str
    episodes: int
    wins: int
    win_rate: float
    per_deck: dict[str, dict[str, float | int]]
    seed: int
    evaluated_at: str


@dataclass
class Entry:
    schema_version: int
    entry_id: str
    checkpoint_path: str
    checkpoint_hash: str
    added_at: str
    panel_version: int
    features: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    panel_results: list[PanelMatchResult] = field(default_factory=list)
    pairwise_results: list[PairwiseMatchResult] = field(default_factory=list)


def entry_id_for(checkpoint_path: Path | str) -> str:
    """Symlinks are NOT resolved — ``checkpoint_latest.pt`` keeps the ``latest``
    suffix even when it points at e.g. ``checkpoint_500.pt``. This is what lets
    a "rolling latest" entry stay distinct from any numbered snapshot."""
    p = Path(checkpoint_path)
    match = _CHECKPOINT_RE.match(p.name)
    if not match:
        raise ValueError(
            f"checkpoint filename must match 'checkpoint_<n|latest>.pt', got {p.name!r}"
        )
    if not p.parent.name:
        raise ValueError(
            f"checkpoint path must include a run-directory parent, got {str(checkpoint_path)!r}"
        )
    return f"{p.parent.name}_{match.group('suffix')}"


def compute_checkpoint_hash(checkpoint_path: Path | str) -> str:
    """SHA-256 of the checkpoint file contents, prefixed with ``sha256:``."""
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _filter_known_kwargs(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Drop dict keys that aren't fields of ``cls`` so future additions degrade gracefully."""
    keep = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in keep}


def _entry_from_dict(d: dict[str, Any]) -> Entry:
    panel = [
        PanelMatchResult(**_filter_known_kwargs(PanelMatchResult, r))
        for r in d.get("panel_results", [])
    ]
    pairwise = [
        PairwiseMatchResult(**_filter_known_kwargs(PairwiseMatchResult, r))
        for r in d.get("pairwise_results", [])
    ]
    return Entry(
        schema_version=d["schema_version"],
        entry_id=d["entry_id"],
        checkpoint_path=d["checkpoint_path"],
        checkpoint_hash=d["checkpoint_hash"],
        added_at=d["added_at"],
        panel_version=d["panel_version"],
        features=d["features"],
        tags=list(d.get("tags", [])),
        panel_results=panel,
        pairwise_results=pairwise,
    )


def write_entry(path: Path | str, entry: Entry) -> None:
    atomic_write_text(
        path, json.dumps(dataclasses.asdict(entry), indent=2, sort_keys=True)
    )


def read_entry(path: Path | str) -> Entry:
    """Load an entry file."""
    return _entry_from_dict(json.loads(Path(path).read_text()))
