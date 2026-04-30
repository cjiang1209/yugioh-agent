"""Tests for index.md rendering (Markdown output)."""

from __future__ import annotations

from yugioh_leaderboard.entry import Entry, PanelMatchResult
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


def _entry(eid: str, rnn: str, seed: int, vs_random: float, vs_greedy: float,
           panel_version: int = 1) -> Entry:
    return Entry(
        schema_version=1,
        entry_id=eid,
        checkpoint_path=f"checkpoints/{eid}.pt",
        checkpoint_hash="sha256:abc",
        added_at="2026-04-11T00:00:00Z",
        panel_version=panel_version,
        features={"rnn_type": rnn, "seed": seed, "reward_shaping": True},
        tags=[],
        panel_results=[
            PanelMatchResult(
                opponent_label="random", episodes=100, wins=int(vs_random * 100),
                win_rate=vs_random, per_deck={}, seed=1, evaluated_at="...",
            ),
            PanelMatchResult(
                opponent_label="greedy", episodes=100, wins=int(vs_greedy * 100),
                win_rate=vs_greedy, per_deck={}, seed=2, evaluated_at="...",
            ),
        ],
        pairwise_results=[],
    )


def test_empty_entries_renders_no_table():
    md = render_index(entries=[], panel=_panel())
    assert "# Yu-Gi-Oh RL Leaderboard" in md
    assert "No entries yet" in md


def test_entries_table_includes_all_panel_opponents():
    entries = [
        _entry("e1", "lstm", 42, 0.95, 0.62),
        _entry("e2", "none", 42, 0.91, 0.55),
    ]
    md = render_index(entries=entries, panel=_panel())
    assert "vs random" in md
    assert "vs greedy" in md
    assert "e1" in md
    assert "e2" in md
    assert "0.95" in md
    assert "0.62" in md


def test_stale_entries_in_separate_section():
    entries = [
        _entry("current", "lstm", 42, 0.9, 0.5, panel_version=2),
        _entry("stale", "none", 42, 0.8, 0.4, panel_version=1),
    ]
    md = render_index(entries=entries, panel=_panel(version=2))
    assert "Stale entries" in md
    main, _, stale = md.partition("Stale entries")
    assert "stale" in stale
    assert "current" in main
