"""Integration test: train a tiny checkpoint, run the full add pipeline.

Skips when libocgcore / cards.cdb / torch / sentence-transformers not
present — same skip pattern as ``tests/rl/test_resume.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.rl.test_resume import _make_checkpoint

from yugioh_rl.config import TrainingConfig

from yugioh_leaderboard.entry import compute_checkpoint_hash
from yugioh_leaderboard.panel import (
    PanelConfig,
    PanelEntry,
    PanelMatchOptions,
)
from yugioh_leaderboard.score import score_checkpoint


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
