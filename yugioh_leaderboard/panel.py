"""Panel config: versioned reference panel + match defaults.

The panel config lives at ``leaderboard/leaderboard.config.json`` and is
edited by hand. Each entry records the ``panel_version`` it was scored
against; entries scored against an older version are flagged "stale" and
excluded from comparisons unless ``--include-stale`` is passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PanelEntry:
    label: str
    spec: str  # "random" | "greedy" | "model:<path>"


@dataclass
class PanelMatchOptions:
    episodes: int
    agent_player: str  # "first" | "second" | "random"
    device: str  # "cpu" | "cuda" | "auto"


@dataclass
class PanelConfig:
    schema_version: int
    panel_version: int
    panel: list[PanelEntry]
    match: PanelMatchOptions
    history: list[dict] = field(default_factory=list)


_VALID_AGENT_PLAYER = ("first", "second", "random")
_VALID_DEVICE = ("cpu", "cuda", "auto")


def _validate_spec(spec: str) -> None:
    if spec in ("random", "greedy"):
        return
    if spec.startswith("model:"):
        if not spec[len("model:"):]:
            raise ValueError("model: panel entry must include checkpoint path")
        return
    raise ValueError(f"unknown opponent kind in panel spec: {spec!r}")


def _require_keys(d: dict, keys: tuple[str, ...], where: str) -> None:
    for k in keys:
        if k not in d:
            raise ValueError(f"{where} missing required key: {k!r}")


def load_panel_config(path: Path | str) -> PanelConfig:
    """Parse and validate a panel config file."""
    raw = json.loads(Path(path).read_text())

    _require_keys(raw, ("schema_version", "panel_version", "panel", "match"), "panel config")

    if not raw["panel"]:
        raise ValueError("panel config 'panel' must contain at least one opponent")

    panel = []
    seen_labels: set[str] = set()
    for i, item in enumerate(raw["panel"]):
        if "label" not in item or "spec" not in item:
            raise ValueError(f"panel[{i}] must have 'label' and 'spec' keys")
        if item["label"] in seen_labels:
            raise ValueError(f"panel[{i}]: duplicate label {item['label']!r}")
        seen_labels.add(item["label"])
        _validate_spec(item["spec"])
        panel.append(PanelEntry(label=item["label"], spec=item["spec"]))

    match_raw = raw["match"]
    _require_keys(match_raw, ("episodes", "agent_player", "device"), "panel.match")
    agent_player = str(match_raw["agent_player"])
    if agent_player not in _VALID_AGENT_PLAYER:
        raise ValueError(
            f"panel.match.agent_player must be one of {_VALID_AGENT_PLAYER}, got {agent_player!r}"
        )
    device = str(match_raw["device"])
    if device not in _VALID_DEVICE:
        raise ValueError(
            f"panel.match.device must be one of {_VALID_DEVICE}, got {device!r}"
        )
    match = PanelMatchOptions(
        episodes=int(match_raw["episodes"]),
        agent_player=agent_player,
        device=device,
    )

    return PanelConfig(
        schema_version=int(raw["schema_version"]),
        panel_version=int(raw["panel_version"]),
        panel=panel,
        match=match,
        history=list(raw.get("history", [])),
    )
