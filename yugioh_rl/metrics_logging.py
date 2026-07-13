"""Non-invasive metrics logging: pure event objects + fan-out sink layer.

Core training/eval code computes ``ScalarMetrics`` / ``CheckpointEvent`` and
hands them to a ``MultiSink``; all logging I/O lives in the sinks. This module
is the only place ``mlflow`` and ``SummaryWriter`` are imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ScalarMetrics:
    """A measurement at a training step. Run-level."""

    scalars: dict[str, float]
    global_step: int


@dataclass
class CheckpointRef:
    """Identity + metadata for a checkpoint file. sha256 is computed lazily by
    the MLflow sink; TensorBoard never needs it."""

    path: Path
    update: int
    global_step: int
    tags: dict[str, str]


@dataclass
class CheckpointEvent:
    """A checkpoint's existence plus optionally metrics measured against it.

    Empty ``scalars`` => pure registration (training). Filled => eval
    measurement (win-rates + per-deck).
    """

    ref: CheckpointRef
    scalars: dict[str, float] = field(default_factory=dict)


class LogSink(Protocol):
    def handle(self, event: ScalarMetrics | CheckpointEvent) -> None: ...

    def close(self) -> None: ...


class MultiSink:
    """Fan-out to a list of sinks. Fail-loud: no exception is swallowed."""

    def __init__(self, sinks: list[LogSink]) -> None:
        self._sinks = sinks

    def handle(self, event: ScalarMetrics | CheckpointEvent) -> None:
        for sink in self._sinks:
            sink.handle(event)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def compute_update_metrics(
    *,
    global_step: int,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    fps: float,
    episode_reward_mean: float | None = None,
    episode_win_rate: float | None = None,
    episode_length_mean: float | None = None,
    deck_win_rates: dict[str, float] | None = None,
    elo: dict | None = None,
    async_stats: dict | None = None,
) -> ScalarMetrics:
    """Build the per-update training ScalarMetrics. Pure; no I/O.

    Key names are byte-identical to the previous inline ``add_scalar`` calls.
    """
    scalars: dict[str, float] = {
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy,
        "perf/fps": fps,
    }
    if episode_reward_mean is not None:
        scalars["episode/reward"] = episode_reward_mean
        scalars["episode/win_rate"] = episode_win_rate
        scalars["episode/length"] = episode_length_mean
    if deck_win_rates:
        for stem, wr in deck_win_rates.items():
            scalars[f"episode/win_rate_deck_{stem}"] = wr
    if elo is not None:
        scalars["selfplay/elo_agent"] = elo["agent"]
        scalars["selfplay/elo_pool_mean"] = elo["pool_mean"]
        scalars["selfplay/elo_pool_min"] = elo["pool_min"]
        scalars["selfplay/elo_pool_max"] = elo["pool_max"]
        scalars["selfplay/occupied"] = elo["occupied"]
    if async_stats is not None:
        scalars["async/version_lag_mean"] = async_stats["version_lag_mean"]
        scalars["async/rollouts_discarded"] = async_stats["rollouts_discarded"]
        if "queue_depth" in async_stats:
            scalars["async/queue_depth"] = async_stats["queue_depth"]
    return ScalarMetrics(scalars=scalars, global_step=global_step)


def flatten_eval(row: dict, label: str) -> dict[str, float]:
    """Flatten a stored eval ``row`` into a scalars dict keyed by opponent label.

    Keys have no ``eval/`` prefix — the TensorBoard sink adds it so MLflow model
    metrics stay unprefixed.
    """
    out: dict[str, float] = {f"win_rate_vs_{label}": row["win_rate"]}
    for stem, d in row["per_deck"].items():
        out[f"win_rate_vs_{label}_deck_{stem}"] = d["win_rate"]
    return out
