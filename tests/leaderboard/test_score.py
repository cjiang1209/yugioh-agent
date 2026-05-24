"""Integration test: train a tiny checkpoint, run the full add pipeline.

Skips when libocgcore / cards.cdb / torch / sentence-transformers not
present — same skip pattern as ``tests/rl/test_resume.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.rl.test_resume import _make_checkpoint
from yugioh_leaderboard.entry import compute_checkpoint_hash
from yugioh_leaderboard.panel import (
    PanelConfig,
    PanelEntry,
    PanelMatchOptions,
)
from yugioh_leaderboard.score import score_checkpoint
from yugioh_rl.config import TrainingConfig


def _is_engine_available() -> bool:
    db_path = Path("assets/cards.cdb")
    if not db_path.exists():
        return False
    try:
        from yugioh_env.lib_loader import load_library

        load_library()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _is_engine_available(),
    reason="libocgcore or assets/cards.cdb missing — skipping integration test",
)


def test_score_checkpoint_writes_panel_results(tmp_path):
    ckpt_dir = tmp_path / "20260411_100000_seed42"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "checkpoint_latest.pt"

    config = TrainingConfig(
        num_envs=1,
        deck_paths=["assets/decks/blue_eyes.ydk"],
        total_timesteps=100,
        seed=42,
    )
    _make_checkpoint(str(ckpt_path), config=config)

    panel = PanelConfig(
        schema_version=1,
        panel_version=1,
        panel=[PanelEntry(label="random", spec="random")],
        match=PanelMatchOptions(episodes=2, agent_player="random", device="cpu"),
        history=[],
    )

    entry = score_checkpoint(ckpt_path, panel)

    assert entry.entry_id == "20260411_100000_seed42_latest"
    assert entry.checkpoint_hash == compute_checkpoint_hash(ckpt_path)
    assert entry.panel_version == 1
    assert len(entry.panel_results) == 1
    assert entry.panel_results[0].opponent_label == "random"
    assert entry.panel_results[0].episodes == 2
    assert 0 <= entry.panel_results[0].win_rate <= 1
    assert entry.features["seed"] == 42
    assert entry.features["card_embeddings"] == "symbolic"


def _make_panel_for_parity(episodes: int) -> PanelConfig:
    """2-opponent panel used by both parity tests below."""
    return PanelConfig(
        schema_version=1,
        panel_version=1,
        panel=[
            PanelEntry(label="random", spec="random"),
            PanelEntry(label="greedy", spec="greedy"),
        ],
        match=PanelMatchOptions(episodes=episodes, agent_player="random", device="cpu"),
        history=[],
    )


def _assert_panel_results_byte_equal(a, b) -> None:
    assert len(a.panel_results) == len(b.panel_results)
    for ra, rb in zip(a.panel_results, b.panel_results, strict=True):
        assert ra.opponent_label == rb.opponent_label
        assert ra.episodes == rb.episodes
        assert ra.wins == rb.wins
        assert ra.win_rate == rb.win_rate
        assert ra.per_deck == rb.per_deck


def _make_parity_checkpoint(tmp_path) -> Path:
    ckpt_dir = tmp_path / "20260411_100000_seed42"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "checkpoint_latest.pt"
    config = TrainingConfig(
        num_envs=1,
        deck_paths=["assets/decks/blue_eyes.ydk"],
        total_timesteps=100,
        seed=42,
    )
    _make_checkpoint(str(ckpt_path), config=config)
    return ckpt_path


def test_score_checkpoint_parallel_matches_sequential(tmp_path):
    """workers=1 vs workers=2 against the same checkpoint must produce
    byte-equal panel results (wins / episodes / per_deck / win_rate).
    Locks the parity guarantee for opponent-axis parallelism.
    """
    ckpt_path = _make_parity_checkpoint(tmp_path)
    panel = _make_panel_for_parity(episodes=4)

    seq_entry = score_checkpoint(ckpt_path, panel, workers=1)
    par_entry = score_checkpoint(ckpt_path, panel, workers=2)

    _assert_panel_results_byte_equal(seq_entry, par_entry)


def test_score_checkpoint_workers_4_matches_workers_2(tmp_path):
    """workers=2 vs workers=4 against the same checkpoint must agree —
    the strongest determinism check, since it varies shard granularity
    on top of opponent-axis parallelism.

    Combined with ``test_score_checkpoint_parallel_matches_sequential``
    (workers=1 vs 2), this transitively pins workers=1 vs 4 — sufficient
    because ``_aggregate_partials`` sorts by ``episode_idx`` regardless
    of worker count, so the byte-equality invariant is structurally the
    same for any K.
    """
    ckpt_path = _make_parity_checkpoint(tmp_path)
    panel = _make_panel_for_parity(episodes=8)

    par2 = score_checkpoint(ckpt_path, panel, workers=2)
    par4 = score_checkpoint(ckpt_path, panel, workers=4)

    _assert_panel_results_byte_equal(par2, par4)
