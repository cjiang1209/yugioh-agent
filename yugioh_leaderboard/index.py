"""Render leaderboard/index.md from entry files.

Sections:
  1. Entries table — sorted by win-rate vs the *last* panel opponent.
  2. Group comparisons — added in Phase 3.
  3. Stale entries — entries scored against an older panel_version.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from yugioh_leaderboard.compare import compare_groups, format_comparison_table
from yugioh_leaderboard.entry import Entry, atomic_write_text, now_iso
from yugioh_leaderboard.features import GROUPING_FIELDS
from yugioh_leaderboard.panel import PanelConfig


def _winrates_by_label(entry: Entry) -> dict[str, float]:
    return {r.opponent_label: r.win_rate for r in entry.panel_results}


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _render_entries_section(entries: list[Entry], panel: PanelConfig) -> str:
    if not entries:
        return "No entries yet — run `scripts/leaderboard.sh add <checkpoint.pt>`.\n"

    feature_cols = ("rnn_type", "reward_shaping", "seed")
    panel_labels = [p.label for p in panel.panel]
    sort_label = panel_labels[-1]

    wr_by_entry = {e.entry_id: _winrates_by_label(e) for e in entries}

    def sort_key(e: Entry) -> float:
        wr = wr_by_entry[e.entry_id].get(sort_label)
        return -1.0 if wr is None else -wr

    sorted_entries = sorted(entries, key=sort_key)
    headers = ["entry_id", *feature_cols, *(f"vs {l}" for l in panel_labels)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for e in sorted_entries:
        row = [e.entry_id]
        for col in feature_cols:
            row.append(str(e.features.get(col, "—")))
        wrs = wr_by_entry[e.entry_id]
        for label in panel_labels:
            row.append(_format_pct(wrs.get(label)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _render_stale_section(stale: list[Entry], panel: PanelConfig) -> str:
    lines = [
        "| entry_id | scored against | current panel |",
        "|---|---|---|",
    ]
    for e in stale:
        lines.append(f"| {e.entry_id} | v{e.panel_version} | v{panel.panel_version} |")
    return "\n".join(lines) + "\n"


def _render_group_sections(fresh: list[Entry]) -> list[str]:
    sections: list[str] = []
    for field_name in GROUPING_FIELDS:
        if field_name == "seed":
            continue
        result = compare_groups(fresh, by_field=field_name)
        if result.skip_reason:
            continue
        sections.append(f"### By {field_name}\n{format_comparison_table(result)}")
    return sections


def render_index(entries: Iterable[Entry], panel: PanelConfig) -> str:
    """Render the full index.md content as a single string."""
    entries = list(entries)
    fresh = [e for e in entries if e.panel_version == panel.panel_version]
    stale = [e for e in entries if e.panel_version < panel.panel_version]

    panel_summary = ", ".join(p.label for p in panel.panel)
    parts = [
        "# Yu-Gi-Oh RL Leaderboard",
        f"_Generated: {now_iso()}_  ·  _Panel v{panel.panel_version}: {panel_summary}_",
        "",
        f"## Entries (sorted by win_rate vs {panel.panel[-1].label})",
        _render_entries_section(fresh, panel),
    ]

    group_sections = _render_group_sections(fresh)
    if group_sections:
        parts.append("## Group comparisons (auto-generated where ≥2 seeds available)")
        parts.extend(group_sections)

    if stale:
        parts.append("## Stale entries (panel version mismatch)")
        parts.append(_render_stale_section(stale, panel))
    return "\n".join(parts)


def write_index_file(path: Path | str, entries: Iterable[Entry], panel: PanelConfig) -> None:
    atomic_write_text(path, render_index(entries, panel))
