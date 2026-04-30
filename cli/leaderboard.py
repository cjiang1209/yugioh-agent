"""Leaderboard CLI: ``add`` / ``compare`` / ``pairwise`` / ``refresh-index``.

Single-user / single-process assumption: no file locks. Don't run two
``add`` commands in parallel — both will write ``index.md`` last-writer-wins
(entry files are safe; they have unique filenames).
"""

from __future__ import annotations

import argparse
import sys
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

    add_p = sub.add_parser("add", help="Score a checkpoint against the panel and write an entry file.")
    add_p.add_argument("checkpoint_path", type=str, help="Path to a checkpoint_<n|latest>.pt file under a run directory.")
    add_p.add_argument("--tags", nargs="+", default=None,
                       help="Replace the entry's tags with these values. Omit to preserve existing tags.")
    add_p.add_argument("--clear-tags", action="store_true",
                       help="Remove all tags from the entry (overrides --tags).")
    add_p.add_argument("--episodes", type=int, default=None,
                       help="Override panel.match.episodes for this score.")
    add_p.add_argument("--seed", type=int, default=None,
                       help="Override the deterministic per-(entry, opponent) seed.")
    add_p.add_argument("--decks", nargs="+", default=None,
                       help="Override the deck pool from the checkpoint's training config.")
    add_p.add_argument("--force", action="store_true",
                       help="Re-score and overwrite even when an entry with matching hash exists.")

    sub.add_parser("refresh-index", help="Regenerate leaderboard/index.md from entry files.")

    return parser


def _validate_subcommand_args(ns: argparse.Namespace) -> None:
    if ns.command == "add":
        if ns.episodes is not None and ns.episodes < 1:
            fatal(f"--episodes: must be >= 1, got {ns.episodes}")
        if not Path(ns.checkpoint_path).exists():
            fatal(f"checkpoint not found: {ns.checkpoint_path}")
        if ns.decks is not None:
            validate_deck_paths(ns.decks, "--decks")


def _load_panel():
    from yugioh_leaderboard.panel import load_panel_config
    if not PANEL_PATH.exists():
        fatal(
            f"panel config not found at {PANEL_PATH}. "
            "Create it first — see docs/superpowers/specs/2026-04-29-rl-feature-evaluation-design.md §6.1"
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
    )
    write_entry(entry_path, entry)
    _refresh_index_file(panel=panel)
    print(f"wrote entry: {entry_path}")
    return 0


def _cmd_refresh_index(ns: argparse.Namespace) -> int:
    _refresh_index_file()
    print(f"wrote index: {INDEX_PATH}")
    return 0


_DISPATCH = {
    "add": _cmd_add,
    "refresh-index": _cmd_refresh_index,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    _validate_subcommand_args(ns)
    return _DISPATCH[ns.command](ns)


if __name__ == "__main__":
    sys.exit(main())
