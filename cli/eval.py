"""Standalone evaluation CLI.

Compares one agent against one or more opponents over a fixed number of
episodes, without spinning up a training loop. Useful for measuring training
progress between checkpoints, sanity-checking heuristics, and picking the
better of two models.

Examples::

    scripts/eval.sh --agent greedy --opponents random greedy --episodes 100 \\
        --deck-paths assets/decks/blue_eyes.ydk

    scripts/eval.sh --agent model:checkpoints/v1/latest.pt \\
        --opponents model:checkpoints/v2/latest.pt --episodes 500 \\
        --agent-player random --json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cli.utils import (
    DEVICE_CHOICES,
    fatal,
    resolve_device,
    validate_deck_paths,
    validate_opponent_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one agent against one or more opponents."
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent spec: 'random', 'greedy', or 'model:path/to/checkpoint.pt'.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        required=True,
        help="One or more opponent specs (same format as --agent).",
    )
    parser.add_argument(
        "--deck-paths",
        nargs="+",
        required=True,
        help="One or more .ydk deck files; agent and opponent sample from this pool each episode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per opponent (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--agent-player",
        type=str,
        default="random",
        choices=["first", "second", "random"],
        help="Agent turn order per episode (default: random).",
    )
    parser.add_argument(
        "--deck-allocation",
        choices=["random", "balanced"],
        default="random",
        help="Per-episode deck assignment (default: random draw; "
        "balanced round-robins for uniform per-deck coverage).",
    )
    parser.add_argument(
        "--mirror-decks",
        action="store_true",
        help="Both players use the same decklist each episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=DEVICE_CHOICES,
        help="Device for both agent-side and env-side model opponents (default: cpu).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for parallel evaluation (default: 1, "
            "sequential). Each worker re-instantiates the agent locally; "
            "results are byte-equal across worker counts."
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="Optional path to write results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    if args.episodes < 0:
        fatal(f"--episodes: must be >= 0, got {args.episodes}")
    if args.workers < 1:
        fatal(f"--workers: must be >= 1, got {args.workers}")
    validate_deck_paths(args.deck_paths)
    validate_opponent_spec(args.agent, "--agent")
    for spec in args.opponents:
        validate_opponent_spec(spec, "--opponents")


def _build_rows(results: list, deck_stems: list[str]) -> list[dict]:
    """Normalize EvalResults into plain dicts for both console + JSON output."""
    from yugioh_rl.eval import eval_result_to_row

    return [{"label": r.opponent_label, **eval_result_to_row(r, deck_stems)} for r in results]


def _print_table(rows: list[dict]) -> None:
    for row in rows:
        print(f"vs {row['label']}: {row['wins']}/{row['episodes']} ({row['win_rate'] * 100:.1f}%)")
        for deck_name, d in row["per_deck"].items():
            print(f"  deck {deck_name}: {d['wins']}/{d['episodes']} ({d['win_rate'] * 100:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = resolve_device(args.device)

    # Deferred so a bad spec exits before paying torch / yugioh_env import cost.
    from yugioh_rl.env_wrapper import parse_deck_pool
    from yugioh_rl.eval import evaluate

    deck_pool = parse_deck_pool(args.deck_paths)

    results = evaluate(
        agent_spec=args.agent,
        deck_pool=deck_pool,
        opponent_specs=args.opponents,
        num_episodes=args.episodes,
        seed=args.seed,
        agent_player=args.agent_player,
        opponent_device=device,
        agent_device=device,
        workers=args.workers,
        deck_allocation=args.deck_allocation,
        mirror_decks=args.mirror_decks,
    )

    deck_stems = [Path(p).stem for p in args.deck_paths]
    rows = _build_rows(results, deck_stems)
    _print_table(rows)

    if args.json:
        Path(args.json).write_text(json.dumps({"opponents": rows}, indent=2))
        print(f"\nWrote JSON results to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
