"""Non-invasive metrics logging: pure event objects + fan-out sink layer.

Core training/eval code computes ``ScalarMetrics`` / ``CheckpointEvent`` and
hands them to a ``MultiSink``; all logging I/O lives in the sinks. This module
is the only place ``mlflow`` and ``SummaryWriter`` are imported.
"""

from __future__ import annotations

import hashlib
import os
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
    """A checkpoint file plus the metadata a sink records alongside it.

    ``params`` are immutable defining facts about the checkpoint (e.g. seed,
    config signature); ``tags`` are optional mutable metadata.
    """

    path: Path
    update: int
    global_step: int
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CheckpointEvent:
    """A checkpoint's existence plus optionally metrics measured against it.

    Empty ``scalars`` => pure registration (training). Filled => eval
    measurement (win-rates + per-deck).
    """

    ref: CheckpointRef
    scalars: dict[str, float] = field(default_factory=dict)


class LogSink(Protocol):
    """A logging destination: ``handle`` records one event; ``close`` flushes
    and releases the underlying resource."""

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
    episode_steps_mean: float | None = None,
    deck_win_rates: dict[str, float] | None = None,
    elo: dict | None = None,
    async_stats: dict | None = None,
    episode_timeout_count: int | None = None,
) -> ScalarMetrics:
    """Build the per-update training ScalarMetrics. Pure; no I/O.

    The scalar keys are the sink-facing metric names.
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
        scalars["episode/steps"] = episode_steps_mean
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
    if episode_timeout_count is not None:
        scalars["episode/timeouts"] = episode_timeout_count
    return ScalarMetrics(scalars=scalars, global_step=global_step)


def flatten_eval(row: dict, label: str) -> dict[str, float]:
    """Flatten a stored eval ``row`` into scalars keyed ``<metric>/<opponent>/<sub>``.

    The opponent sits in the middle so both UIs section usefully: TensorBoard
    groups on the first ``/`` (``<metric>``), MLflow on the last ``/`` (so
    ``<metric>/<opponent>`` becomes one card with the sub-series overlaid, e.g.
    win_rate overall/play_first/play_second together per opponent). No ``eval/``
    prefix. A win-rate split is emitted only when it had episodes; missing
    (old-row) fields are skipped.
    """
    out: dict[str, float] = {f"win_rate/{label}/overall": row["win_rate"]}
    for stem, d in row.get("per_deck", {}).items():
        out[f"win_rate_by_deck/{label}/{stem}"] = d["win_rate"]
    for metric in ("steps", "turns"):
        stats = row.get(metric)
        if stats:
            for sub in ("mean", "std", "median", "max"):
                out[f"{metric}/{label}/{sub}"] = stats[sub]
    if "play_first_rate" in row:
        out[f"play_first_rate/{label}"] = row["play_first_rate"]
    if row.get("episodes_first"):
        out[f"win_rate/{label}/play_first"] = row["wins_first"] / row["episodes_first"]
    if row.get("episodes_second"):
        out[f"win_rate/{label}/play_second"] = row["wins_second"] / row["episodes_second"]
    if "timeouts" in row:
        out[f"timeouts/{label}"] = row["timeouts"]
    return out


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file's bytes (read in chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TensorBoardSink:
    """Writes events to a TensorBoard SummaryWriter.

    All scalar keys are written verbatim: ScalarMetrics at ``event.global_step``,
    CheckpointEvent scalars at the checkpoint's ``global_step``.
    """

    def __init__(self, writer) -> None:
        self._writer = writer

    def handle(self, event: ScalarMetrics | CheckpointEvent) -> None:
        if isinstance(event, ScalarMetrics):
            for key, value in event.scalars.items():
                self._writer.add_scalar(key, value, event.global_step)
        elif isinstance(event, CheckpointEvent):
            for key, value in event.scalars.items():
                self._writer.add_scalar(key, value, event.ref.global_step)

    def close(self) -> None:
        self._writer.close()


class MLflowSink:
    """Writes events to MLflow, one external LoggedModel per checkpoint keyed by
    the sha256 of its file: a registration event (no scalars) creates the model
    and uploads the ``.pt``; an eval event finds that model by hash and attaches
    its win-rate metrics. Run-level ScalarMetrics go to the active run.

    ``mlflow_module`` is injected so the sink is testable with a fake; a run
    must already be active (opened by the ``build_*_sinks`` factory).
    """

    def __init__(self, mlflow_module) -> None:
        self._mlflow = mlflow_module
        self._hash_cache: dict[Path, str] = {}

    def handle(self, event: ScalarMetrics | CheckpointEvent) -> None:
        if isinstance(event, ScalarMetrics):
            self._mlflow.log_metrics(event.scalars, step=event.global_step)
        elif isinstance(event, CheckpointEvent):
            self._handle_checkpoint(event)

    def _handle_checkpoint(self, event: CheckpointEvent) -> None:
        ref = event.ref
        # Cache by path: a sweep handles the same checkpoint once per opponent,
        # and the file is immutable — hash it once, not once per pair.
        sha = self._hash_cache.get(ref.path)
        if sha is None:
            sha = self._hash_cache[ref.path] = sha256_file(ref.path)
        hits = self._mlflow.search_logged_models(
            filter_string=f"params.checkpoint_hash='{sha}'",
            output_format="list",
        )
        if hits:
            model = hits[0]
        else:
            model = self._mlflow.create_external_model(
                name=f"checkpoint_{ref.update}",
                tags=dict(ref.tags),
                params={
                    **ref.params,
                    "checkpoint_hash": sha,
                    "update": str(ref.update),
                    "global_step": str(ref.global_step),
                },
            )
            self._mlflow.log_artifact(
                str(ref.path),
                artifact_path=f"checkpoints/checkpoint_{ref.update}",
            )
        if event.scalars:
            self._mlflow.log_metrics(event.scalars, step=ref.global_step, model_id=model.model_id)

    def close(self) -> None:
        self._mlflow.end_run()


EXPERIMENT_NAME = "yugioh"


def _import_mlflow():
    """Import mlflow, converting a missing dependency into an actionable error."""
    try:
        import mlflow
    except ImportError as e:
        raise RuntimeError(
            "--log-to mlflow requires the 'mlflow' package. Install it with "
            "`pip install -e '.[train]'` (the train extra includes mlflow)."
        ) from e
    return mlflow


def _require_tracking_uri(mlflow_module) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "--log-to mlflow requires the MLFLOW_TRACKING_URI environment "
            "variable (e.g. http://127.0.0.1:5000)."
        )
    mlflow_module.set_tracking_uri(uri)


def _new_tb_writer(log_dir: str, purge_step: int | None):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=log_dir, purge_step=purge_step)


def _open_experiment():
    """Import mlflow, configure the tracking URI + experiment, and enable
    hardware telemetry (CPU/mem/disk/GPU as system/* metrics). Returns the
    module, ready for the caller to start its run."""
    mlflow = _import_mlflow()
    _require_tracking_uri(mlflow)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.enable_system_metrics_logging()
    return mlflow


def build_training_sinks(
    *,
    log_to: list[str],
    save_dir: str,
    purge_step: int | None,
    params: dict[str, str],
) -> MultiSink:
    """Build the sink fan-out for a training run.

    MLflow: reattaches the run recorded in ``<save_dir>/mlflow_run_id.txt`` if
    present (continuous curve across --resume), else starts a fresh named run
    and persists its id. Logs ``params`` once, and uploads the run's
    ``config.json`` snapshot as an artifact (the authoritative, structured
    record — ``params`` are the flattened/stringified view).
    """
    sinks: list[LogSink] = []
    if "tensorboard" in log_to:
        writer = _new_tb_writer(str(Path(save_dir) / "logs"), purge_step)
        sinks.append(TensorBoardSink(writer))
    if "mlflow" in log_to:
        mlflow = _open_experiment()
        id_file = Path(save_dir) / "mlflow_run_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else None
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(run_name=f"train_{Path(save_dir).name}")
            id_file.write_text(run.info.run_id)
            mlflow.log_params(params)
        # Upload the config snapshot (structured; safe to re-log on resume so it
        # reflects the effective config after any allowlisted overrides).
        config_json = Path(save_dir) / "config.json"
        if config_json.exists():
            mlflow.log_artifact(str(config_json))
        sinks.append(MLflowSink(mlflow))
    return MultiSink(sinks)


def build_eval_sinks(
    *,
    log_to: list[str],
    run_dir: str,
    params: dict[str, str],
    subdir: str = "eval",
    run_name: str | None = None,
) -> MultiSink:
    """Build the sink fan-out for an offline eval sweep.

    TensorBoard writes to ``<run_dir>/logs/<subdir>`` and MLflow opens a named run
    ``run_name`` (default ``eval_<run_dir_name>``), so eval win-rate source-runs are
    meaningful, not auto-named orphans. ``subdir``/``run_name`` let step-matched
    evaluation land in its own board + run (e.g. ``eval_vs_<opponent>``) instead of
    the classic ``eval`` board. Eval runs are always fresh — no resume — so params
    are logged unconditionally.
    """
    run_name = run_name or f"eval_{Path(run_dir).name}"
    sinks: list[LogSink] = []
    if "tensorboard" in log_to:
        log_dir = Path(run_dir) / "logs" / subdir
        sinks.append(TensorBoardSink(_new_tb_writer(str(log_dir), None)))
    if "mlflow" in log_to:
        mlflow = _open_experiment()
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        sinks.append(MLflowSink(mlflow))
    return MultiSink(sinks)
