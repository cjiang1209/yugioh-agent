"""Leaderboard CLI: ``add`` / ``compare`` / ``pairwise`` / ``refresh-index``.

Single-user / single-process assumption: no file locks. Don't run two
``add`` commands in parallel — both will write ``index.md`` last-writer-wins
(entry files are safe; they have unique filenames).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

from cli.utils import fatal, validate_deck_paths

LEADERBOARD_DIR = Path("leaderboard")
ENTRIES_DIR = LEADERBOARD_DIR / "entries"
INDEX_PATH = LEADERBOARD_DIR / "index.md"
PANEL_PATH = LEADERBOARD_DIR / "leaderboard.config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaderboard",
        description=(
            "RL feature evaluation leaderboard. "
            "Single-user / single-process — don't run two `add` commands in parallel."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser(
        "add", help="Score a checkpoint against the panel and write an entry file."
    )
    add_p.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a checkpoint_<n|latest>.pt file under a run directory.",
    )
    add_p.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Replace the entry's tags with these values. Omit to preserve existing tags.",
    )
    add_p.add_argument(
        "--clear-tags",
        action="store_true",
        help="Remove all tags from the entry (overrides --tags).",
    )
    add_p.add_argument(
        "--episodes", type=int, default=None, help="Override panel.match.episodes for this score."
    )
    add_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the deterministic per-(entry, opponent) seed.",
    )
    add_p.add_argument(
        "--decks",
        nargs="+",
        default=None,
        help="Override the deck pool from the checkpoint's training config.",
    )
    add_p.add_argument(
        "--force",
        action="store_true",
        help="Re-score and overwrite even when an entry with matching hash exists.",
    )
    add_p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for parallel panel scoring (default: 1, sequential). "
        "Results are byte-equal across worker counts.",
    )

    cmp = sub.add_parser("compare", help="Compare entries grouped by a feature field.")
    cmp.add_argument(
        "--by", type=str, default=None, help="features field to group entries by (e.g. rnn_type)"
    )
    cmp.add_argument(
        "--by-tag", nargs="+", default=None, help="alternative grouping by user-supplied tags"
    )
    cmp.add_argument(
        "--filter",
        nargs="+",
        default=[],
        help="KEY=VALUE filters; entries matching all are included",
    )
    cmp.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help="restrict report to specific panel opponent labels",
    )
    cmp.add_argument(
        "--include-stale",
        action="store_true",
        help="include entries scored against an older panel_version",
    )
    cmp.add_argument(
        "--json", action="store_true", help="emit JSON instead of formatted Markdown table"
    )

    pw = sub.add_parser("pairwise", help="Run a head-to-head match between two entries.")
    pw.add_argument("entry_a_id", type=str, help="entry_id of the first checkpoint")
    pw.add_argument("entry_b_id", type=str, help="entry_id of the second checkpoint")
    pw.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes for the head-to-head match."
    )
    pw.add_argument(
        "--seed", type=int, default=None, help="Override the deterministic per-pair seed."
    )
    pw.add_argument(
        "--decks",
        nargs="+",
        default=None,
        help="Override the deck pool (default: intersection of both entries').",
    )
    pw.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for parallel evaluation (default: 1, sequential). "
        "Single opponent → all parallelism is episode-shard.",
    )

    sub.add_parser("refresh-index", help="Regenerate leaderboard/index.md from entry files.")

    return parser


def _validate_subcommand_args(ns: argparse.Namespace) -> None:
    if ns.command == "add":
        if ns.episodes is not None and ns.episodes < 1:
            fatal(f"--episodes: must be >= 1, got {ns.episodes}")
        if ns.workers < 1:
            fatal(f"--workers: must be >= 1, got {ns.workers}")
        if not Path(ns.checkpoint_path).exists():
            fatal(f"checkpoint not found: {ns.checkpoint_path}")
        if ns.decks is not None:
            validate_deck_paths(ns.decks, "--decks")

    if ns.command == "compare":
        if ns.by is None and ns.by_tag is None:
            fatal("compare: must pass either --by or --by-tag")
        for f in ns.filter:
            if "=" not in f:
                fatal(f"--filter: expected KEY=VALUE, got {f!r}")
            k, v = f.split("=", 1)
            if not k.strip() or not v.strip():
                fatal(f"--filter: KEY and VALUE must be non-empty, got {f!r}")
        if ns.by is not None:
            from yugioh_leaderboard.features import GROUPING_FIELDS

            if ns.by not in GROUPING_FIELDS:
                fatal(
                    f"--by: unknown feature field {ns.by!r}. Available: "
                    + ", ".join(sorted(GROUPING_FIELDS))
                )

    if ns.command == "pairwise":
        if ns.episodes < 1:
            fatal(f"--episodes: must be >= 1, got {ns.episodes}")
        if ns.workers < 1:
            fatal(f"--workers: must be >= 1, got {ns.workers}")
        if ns.decks is not None:
            validate_deck_paths(ns.decks, "--decks")


def _load_panel():
    from yugioh_leaderboard.panel import load_panel_config

    if not PANEL_PATH.exists():
        fatal(
            f"panel config not found at {PANEL_PATH}. "
            "See the Leaderboard System section in CLAUDE.md for the schema."
        )
    return load_panel_config(PANEL_PATH)


def _load_all_entries():
    from yugioh_leaderboard.entry import read_entry

    if not ENTRIES_DIR.exists():
        return []
    return [read_entry(p) for p in sorted(ENTRIES_DIR.glob("*.json"))]


def _refresh_index_file(panel=None):
    from yugioh_leaderboard.index import write_index_file

    if panel is None:
        panel = _load_panel()
    entries = _load_all_entries()
    write_index_file(INDEX_PATH, entries, panel)


def _cmd_add(ns: argparse.Namespace) -> int:
    from yugioh_leaderboard.entry import (
        compute_checkpoint_hash,
        entry_id_for,
        read_entry,
        write_entry,
    )
    from yugioh_leaderboard.score import score_checkpoint

    panel = _load_panel()
    eid = entry_id_for(ns.checkpoint_path)
    entry_path = ENTRIES_DIR / f"{eid}.json"
    chash = compute_checkpoint_hash(ns.checkpoint_path)

    existing = read_entry(entry_path) if entry_path.exists() else None
    if (
        existing is not None
        and not ns.force
        and existing.checkpoint_hash == chash
        and existing.panel_version == panel.panel_version
    ):
        print(
            f"entry already exists with matching hash and panel v{panel.panel_version} "
            "— skipping (use --force to re-run)"
        )
        return 0

    if ns.clear_tags:
        tags = []
    elif ns.tags is not None:
        tags = ns.tags
    else:
        tags = existing.tags if existing else None

    entry = score_checkpoint(
        ns.checkpoint_path,
        panel,
        deck_paths_override=ns.decks,
        episodes_override=ns.episodes,
        seed_override=ns.seed,
        tags=tags,
        existing_entry=existing,
        precomputed_hash=chash,
        workers=ns.workers,
    )
    write_entry(entry_path, entry)
    _refresh_index_file(panel=panel)
    print(f"wrote entry: {entry_path}")
    return 0


def _cmd_compare(ns: argparse.Namespace) -> int:
    import json as _json

    from yugioh_leaderboard.compare import compare_groups, format_comparison_table, matches_filter

    panel = _load_panel()
    entries = _load_all_entries()
    if not ns.include_stale:
        entries = [e for e in entries if e.panel_version == panel.panel_version]

    flt: dict[str, object] = {}
    for raw in ns.filter:
        k, v = raw.split("=", 1)
        flt[k.strip()] = v.strip()

    filtered = [e for e in entries if matches_filter(e, flt or None)]

    deck_sets = {",".join(e.features.get("deck_paths") or []) for e in filtered}
    if len(deck_sets) > 1:
        print(
            "WARNING: comparing across different deck pools — results may not be apples-to-apples",
            file=sys.stderr,
        )
    opponent_set = {e.features.get("training_opponent") for e in filtered}
    if len(opponent_set) > 1 and ns.by != "training_opponent":
        print(
            "WARNING: training_opponent differs across entries — "
            "groups may have trained against different opponents",
            file=sys.stderr,
        )

    if ns.by_tag:
        synthesized = [
            replace(
                e,
                features={
                    **e.features,
                    "__tag_group__": ",".join(sorted(t for t in e.tags if t in ns.by_tag)),
                },
            )
            for e in entries
        ]
        result = compare_groups(
            synthesized, by_field="__tag_group__", filter=flt or None, opponents=ns.opponents
        )
    else:
        result = compare_groups(entries, by_field=ns.by, filter=flt or None, opponents=ns.opponents)

    if ns.json:
        print(_json.dumps(asdict(result), indent=2, default=str))
    else:
        if result.skip_reason:
            print(f"no comparison: {result.skip_reason}")
            return 0
        print(format_comparison_table(result))
    return 0


def _cmd_pairwise(ns: argparse.Namespace) -> int:
    from yugioh_leaderboard.entry import read_entry, write_entry
    from yugioh_leaderboard.pairwise import NoSharedDecksError, run_pairwise

    panel = _load_panel()
    a_path = ENTRIES_DIR / f"{ns.entry_a_id}.json"
    b_path = ENTRIES_DIR / f"{ns.entry_b_id}.json"
    if not a_path.exists():
        fatal(f"entry not found: {ns.entry_a_id}")
    if not b_path.exists():
        fatal(f"entry not found: {ns.entry_b_id}")

    entry_a = read_entry(a_path)
    entry_b = read_entry(b_path)
    for entry in (entry_a, entry_b):
        if not Path(entry.checkpoint_path).exists():
            fatal(f"entry {entry.entry_id} references missing checkpoint: {entry.checkpoint_path}")

    try:
        new_a, new_b = run_pairwise(
            entry_a,
            entry_b,
            panel,
            episodes=ns.episodes,
            seed=ns.seed,
            decks_override=ns.decks,
            workers=ns.workers,
        )
    except NoSharedDecksError as e:
        fatal(str(e))

    write_entry(a_path, new_a)
    write_entry(b_path, new_b)
    _refresh_index_file(panel=panel)
    rec = next(r for r in new_a.pairwise_results if r.vs_entry_id == ns.entry_b_id)
    print(f"{ns.entry_a_id} vs {ns.entry_b_id}: {rec.wins}/{rec.episodes} ({rec.win_rate:.1%})")
    return 0


def _cmd_refresh_index(ns: argparse.Namespace) -> int:
    _refresh_index_file()
    print(f"wrote index: {INDEX_PATH}")
    return 0


_DISPATCH = {
    "add": _cmd_add,
    "compare": _cmd_compare,
    "pairwise": _cmd_pairwise,
    "refresh-index": _cmd_refresh_index,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    _validate_subcommand_args(ns)
    return _DISPATCH[ns.command](ns)


if __name__ == "__main__":
    sys.exit(main())
