"""Offline checkpoint-sweep evaluator.

Evaluates a training run's checkpoints against chosen opponents and writes
win-rate curves to <run-dir>/logs/eval/ keyed by each checkpoint's global_step.
Thin driver over yugioh_rl.eval.evaluate; the trainer is never modified.
"""

from __future__ import annotations

import argparse
import json as _json
import logging
import re
import sys
from pathlib import Path

import torch

from yugioh_rl.eval import (
    eval_result_to_row,
    evaluate,
    opponent_label_from_spec,
)
from yugioh_rl.metrics_logging import (
    CheckpointEvent,
    CheckpointRef,
    build_eval_sinks,
    flatten_eval,
)

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


class SweepError(Exception):
    """Pre-loop fatal condition (no checkpoints / unresolvable deck pool)."""


def checkpoint_update(path: Path) -> int:
    """Parse the integer update number from a checkpoint_<N>.pt filename."""
    m = _CKPT_RE.match(Path(path).name)
    if m is None:
        raise ValueError(f"not a numbered checkpoint: {path}")
    return int(m.group(1))


def discover_checkpoints(run_dir: str, stride: int) -> list[Path]:
    """Return checkpoint_<N>.pt paths, sorted ascending by N, keeping every
    ``stride``th. Non-numeric names (e.g. checkpoint_latest.pt) are excluded."""
    paths = [p for p in Path(run_dir).glob("checkpoint_*.pt") if _CKPT_RE.match(p.name)]
    paths.sort(key=checkpoint_update)
    return paths[::stride]


class Manifest:
    """Idempotency record of completed (update, label) eval pairs.

    On-disk: {"results": [{"update", "label", "result"}, ...]}. Only successful
    pairs are ever recorded; record() flushes immediately so an interrupted
    sweep is always a valid resume point.
    """

    def __init__(self, path: Path, rows: list[dict]) -> None:
        self.path = Path(path)
        # Full entries keyed by pair; dict insertion order preserves row order,
        # so the on-disk {"results": [...]} shape is unchanged.
        self._index = {(r["update"], r["label"]): r for r in rows}

    @classmethod
    def load(cls, path: Path) -> Manifest:
        path = Path(path)
        if not path.exists():
            return cls(path, [])
        data = _json.loads(path.read_text())
        return cls(path, data.get("results", []))

    def has(self, update: int, label: str) -> bool:
        return (update, label) in self._index

    def get(self, update: int, label: str) -> dict | None:
        entry = self._index.get((update, label))
        return entry["result"] if entry is not None else None

    def record(self, update: int, label: str, result_row: dict) -> None:
        self._index[(update, label)] = {
            "update": update,
            "label": label,
            "result": result_row,
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_json.dumps({"results": list(self._index.values())}, indent=2))


def _config_deck_paths(config) -> list[str] | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("deck_paths")
    return getattr(config, "deck_paths", None)


def derive_deck_paths(checkpoints, override, load_fn=torch.load) -> list[str]:
    """Return the eval deck pool paths. Prefer ``override``; else derive from the
    first readable checkpoint's config.deck_paths."""
    if override:
        return list(override)
    for ckpt in checkpoints:
        try:
            data = load_fn(ckpt, map_location="cpu", weights_only=False)
            paths = _config_deck_paths(data.get("config"))
        except Exception:
            continue
        if paths:
            return list(paths)
    raise SweepError("could not derive deck pool from any checkpoint; pass --deck-paths")


def _emit_row(sink, label: str, ckpt: Path, update: int, row: dict, global_step: int) -> None:
    """Emit a per-checkpoint eval measurement (CheckpointEvent) from a result row;
    shared by the replayed-from-manifest and freshly-evaluated paths."""
    sink.handle(
        CheckpointEvent(
            ref=CheckpointRef(path=Path(ckpt), update=update, global_step=global_step),
            scalars=flatten_eval(row, label),
        )
    )


def run_sweep(
    *,
    checkpoints,
    opponents,
    deck_pool,
    deck_paths,
    manifest,
    sink,
    num_episodes,
    seed,
    workers,
    agent_player,
    deck_allocation="random",
    mirror_decks=False,
    force,
    evaluate_fn=evaluate,
    load_fn=torch.load,
) -> dict:
    """Sweep evaluator: for each (checkpoint, opponent) pair, eval or replay.

    Returns dict with keys: ok, failed, skipped, failures.
    """
    deck_stems = [Path(p).stem for p in deck_paths]
    ok = failed = skipped = 0
    failures: list[tuple[int, str]] = []

    for ckpt in checkpoints:
        update = checkpoint_update(ckpt)
        global_step = None  # loaded lazily once per checkpoint, reused across opponents
        for opp in opponents:
            label = opponent_label_from_spec(opp)
            if manifest.has(update, label) and not force:
                # Replay recorded result, using the global_step stored at record time.
                row = manifest.get(update, label)
                gs = row.get("global_step", update)
                _emit_row(sink, label, ckpt, update, row, gs)
                skipped += 1
                continue
            try:
                # Load global_step once per checkpoint (not per opponent).
                if global_step is None:
                    global_step = load_fn(ckpt, map_location="cpu", weights_only=False)[
                        "global_step"
                    ]
                # Evaluate
                results = evaluate_fn(
                    agent_spec=f"model:{ckpt}",
                    deck_pool=deck_pool,
                    opponent_specs=[opp],
                    num_episodes=num_episodes,
                    seed=seed,
                    workers=workers,
                    agent_player=agent_player,
                    deck_allocation=deck_allocation,
                    mirror_decks=mirror_decks,
                )
                # Record in manifest and emit event.
                row = eval_result_to_row(results[0], deck_stems)
                row["global_step"] = global_step
                _emit_row(sink, label, ckpt, update, row, global_step)
                manifest.record(update, label, row)
                ok += 1
            except Exception as e:
                logger.warning("eval failed for checkpoint_%d vs %s: %s", update, label, e)
                failures.append((update, label))
                failed += 1
                continue
    return {"ok": ok, "failed": failed, "skipped": skipped, "failures": failures}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a run's checkpoints against opponents.")
    p.add_argument("--run-dir", required=True, help="Training run dir with checkpoint_*.pt")
    p.add_argument(
        "--opponents",
        required=True,
        nargs="+",
        help="Opponent specs: random / greedy / model:path / ygo-agent:url",
    )
    p.add_argument("--stride", type=int, default=1, help="Evaluate every Nth checkpoint")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent-player", choices=["first", "second", "random"], default="random")
    p.add_argument(
        "--deck-paths",
        nargs="*",
        default=None,
        help="Override eval decks (default: derive from checkpoint config)",
    )
    p.add_argument(
        "--deck-allocation",
        choices=["random", "balanced"],
        default="random",
        help="Per-episode deck assignment (default: random draw; "
        "balanced round-robins for uniform per-deck coverage).",
    )
    p.add_argument(
        "--mirror-decks",
        action="store_true",
        help="Both players use the same decklist each episode.",
    )
    p.add_argument("--force", action="store_true", help="Re-evaluate recorded pairs")
    p.add_argument(
        "--log-to",
        nargs="+",
        choices=["tensorboard", "mlflow"],
        default=["tensorboard"],
        help="Logging destinations (default: tensorboard).",
    )
    p.add_argument("--json", default=None, help="Also write a JSON summary to this path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.stride < 1:
        print(f"--stride must be >= 1, got {args.stride}", file=sys.stderr)
        return 2

    checkpoints = discover_checkpoints(args.run_dir, args.stride)
    if not checkpoints:
        print(f"no checkpoints found in {args.run_dir}", file=sys.stderr)
        return 2

    # Deferred imports (torch/env) so arg errors are cheap.
    from cli.utils import validate_deck_paths, validate_opponent_spec

    from yugioh_rl.env_wrapper import parse_deck_pool

    for opp in args.opponents:
        validate_opponent_spec(opp, "--opponents")  # exits via fatal() on malformed spec

    try:
        deck_paths = derive_deck_paths(checkpoints, args.deck_paths)
    except SweepError as e:
        print(str(e), file=sys.stderr)
        return 2

    validate_deck_paths(deck_paths)  # exits via fatal() on missing / non-.ydk paths
    deck_pool = parse_deck_pool(deck_paths)
    log_dir = Path(args.run_dir) / "logs" / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(log_dir / "manifest.json")
    eval_params = {
        "run_dir": args.run_dir,
        "opponents": ",".join(args.opponents),
        "episodes": str(args.episodes),
        "seed": str(args.seed),
        "workers": str(args.workers),
        "agent_player": args.agent_player,
        "deck_allocation": args.deck_allocation,
        "mirror_decks": str(args.mirror_decks),
        "stride": str(args.stride),
        "decks": ",".join(Path(p).stem for p in deck_paths),
    }
    sink = build_eval_sinks(log_to=args.log_to, run_dir=args.run_dir, params=eval_params)
    try:
        summary = run_sweep(
            checkpoints=checkpoints,
            opponents=args.opponents,
            deck_pool=deck_pool,
            deck_paths=deck_paths,
            manifest=manifest,
            sink=sink,
            num_episodes=args.episodes,
            seed=args.seed,
            workers=args.workers,
            agent_player=args.agent_player,
            deck_allocation=args.deck_allocation,
            mirror_decks=args.mirror_decks,
            force=args.force,
        )
    finally:
        sink.close()

    print(
        f"\nSweep complete: {summary['ok']} ok, {summary['failed']} failed, "
        f"{summary['skipped']} skipped"
    )
    for update, label in summary["failures"]:
        print(f"  FAILED: checkpoint_{update} vs {label}")

    if args.json:
        Path(args.json).write_text(_json.dumps(summary, indent=2))
        print(f"Wrote JSON summary to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
