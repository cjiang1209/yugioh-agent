"""Tests for index.md rendering (Markdown output)."""

from __future__ import annotations

from yugioh_leaderboard.index import render_index
from yugioh_leaderboard.panel import (
    PanelConfig,
    PanelEntry,
    PanelMatchOptions,
)


def _panel(version: int = 1) -> PanelConfig:
    return PanelConfig(
        schema_version=1,
        panel_version=version,
        panel=[
            PanelEntry(label="random", spec="random"),
            PanelEntry(label="greedy", spec="greedy"),
        ],
        match=PanelMatchOptions(episodes=100, agent_player="random", device="cpu"),
        history=[],
    )


def test_empty_entries_renders_no_table():
    md = render_index(entries=[], panel=_panel())
    assert "# Yu-Gi-Oh RL Leaderboard" in md
    assert "No entries yet" in md


def test_entries_table_includes_all_panel_opponents(make_entry):
    entries = [
        make_entry("e1", "lstm", 42, 0.95, 0.62),
        make_entry("e2", "none", 42, 0.91, 0.55),
    ]
    md = render_index(entries=entries, panel=_panel())
    assert "vs random" in md
    assert "vs greedy" in md
    assert "e1" in md
    assert "e2" in md
    assert "0.95" in md
    assert "0.62" in md


def test_stale_entries_in_separate_section(make_entry):
    entries = [
        make_entry("current", "lstm", 42, 0.9, 0.5, panel_version=2),
        make_entry("stale", "none", 42, 0.8, 0.4, panel_version=1),
    ]
    md = render_index(entries=entries, panel=_panel(version=2))
    assert "Stale entries" in md
    main, _, stale = md.partition("Stale entries")
    assert "stale" in stale
    assert "current" in main


def test_group_comparisons_section_renders_when_two_groups(make_entry):
    entries = [
        make_entry("none1", "none", 42, 0.50, 0.40),
        make_entry("none2", "none", 43, 0.50, 0.40),
        make_entry("none3", "none", 44, 0.50, 0.40),
        make_entry("lstm1", "lstm", 42, 0.60, 0.50),
        make_entry("lstm2", "lstm", 43, 0.60, 0.50),
        make_entry("lstm3", "lstm", 44, 0.60, 0.50),
    ]
    md = render_index(entries=entries, panel=_panel())
    assert "Group comparisons" in md
    assert "rnn_type" in md


def test_group_comparisons_section_omitted_with_one_group(make_entry):
    entries = [
        make_entry("a", "lstm", 42, 0.6, 0.5),
        make_entry("b", "lstm", 43, 0.6, 0.5),
    ]
    md = render_index(entries=entries, panel=_panel())
    assert "Group comparisons" not in md
