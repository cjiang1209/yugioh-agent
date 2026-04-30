"""Shared fixtures and factories for leaderboard tests."""

from __future__ import annotations

import pytest

from yugioh_leaderboard.entry import Entry, PanelMatchResult


def _make_entry(
    eid: str,
    rnn: str,
    seed: int,
    vs_random: float,
    vs_greedy: float,
    *,
    panel_version: int = 1,
    extra_features: dict | None = None,
) -> Entry:
    features = {
        "rnn_type": rnn,
        "seed": seed,
        "reward_shaping": True,
        "deck_paths": ["a"],
    }
    if extra_features:
        features.update(extra_features)
    return Entry(
        schema_version=1,
        entry_id=eid,
        checkpoint_path=f"checkpoints/{eid}.pt",
        checkpoint_hash="sha256:abc",
        added_at="2026-04-11T00:00:00Z",
        panel_version=panel_version,
        features=features,
        tags=[],
        panel_results=[
            PanelMatchResult(
                opponent_label="random",
                episodes=100,
                wins=int(vs_random * 100),
                win_rate=vs_random,
                per_deck={},
                seed=1,
                evaluated_at="t",
            ),
            PanelMatchResult(
                opponent_label="greedy",
                episodes=100,
                wins=int(vs_greedy * 100),
                win_rate=vs_greedy,
                per_deck={},
                seed=2,
                evaluated_at="t",
            ),
        ],
        pairwise_results=[],
    )


@pytest.fixture
def make_entry():
    return _make_entry
