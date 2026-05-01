"""Integration: pairwise match mirrors symmetrically + unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yugioh_leaderboard.entry import PairwiseMatchResult
from yugioh_leaderboard.pairwise import (
    NoSharedDecksError,
    _pair_seed,
    replace_or_append_pairwise,
)


def test_no_shared_decks_error_is_value_error_subclass():
    """CLI catches the narrow class, but library callers can still
    `except ValueError` if they prefer the broader catch."""
    assert issubclass(NoSharedDecksError, ValueError)


def _make_record(vs_id: str, wins: int, episodes: int = 10) -> PairwiseMatchResult:
    return PairwiseMatchResult(
        vs_entry_id=vs_id, vs_checkpoint_hash="sha256:x",
        episodes=episodes, wins=wins, win_rate=wins / episodes,
        per_deck={}, seed=1, evaluated_at="t",
    )


def test_replace_or_append_appends_when_empty():
    out = replace_or_append_pairwise([], _make_record("b", 5))
    assert len(out) == 1
    assert out[0].vs_entry_id == "b"


def test_replace_or_append_overwrites_matching_vs_id():
    """Re-running a pair must overwrite the prior record, not duplicate it."""
    initial = [_make_record("b", 3), _make_record("c", 7)]
    out = replace_or_append_pairwise(initial, _make_record("b", 8))
    assert len(out) == 2
    by_id = {r.vs_entry_id: r for r in out}
    assert by_id["b"].wins == 8  # overwritten
    assert by_id["c"].wins == 7  # untouched


def test_pair_seed_is_order_independent():
    assert _pair_seed("a", "b") == _pair_seed("b", "a")


def _engine_available() -> bool:
    if not Path("assets/cards.cdb").exists():
        return False
    try:
        from yugioh_env.lib_loader import load_library
        load_library()
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _engine_available(),
    reason="libocgcore or assets/cards.cdb missing",
)
def test_pairwise_mirrors_symmetrically(tmp_path):
    pytest.importorskip("torch")

    from tests.rl.test_resume import _make_checkpoint
    from yugioh_rl.config import TrainingConfig

    from yugioh_leaderboard.entry import Entry
    from yugioh_leaderboard.features import extract_features
    from yugioh_leaderboard.pairwise import run_pairwise
    from yugioh_leaderboard.panel import (
        PanelConfig,
        PanelEntry,
        PanelMatchOptions,
    )

    def _stub_entry(ckpt: Path, eid: str, cfg: TrainingConfig) -> Entry:
        return Entry(
            schema_version=1, entry_id=eid, checkpoint_path=str(ckpt),
            checkpoint_hash="sha256:placeholder", added_at="t",
            panel_version=1, features=extract_features(cfg),
            tags=[], panel_results=[], pairwise_results=[],
        )

    cfg = TrainingConfig(num_envs=1, deck_paths=["assets/decks/blue_eyes.ydk"], seed=42)
    a_dir = tmp_path / "a_run"; a_dir.mkdir()
    b_dir = tmp_path / "b_run"; b_dir.mkdir()
    a_ckpt = a_dir / "checkpoint_latest.pt"
    b_ckpt = b_dir / "checkpoint_latest.pt"
    _make_checkpoint(str(a_ckpt), config=cfg)
    _make_checkpoint(str(b_ckpt), config=cfg)

    panel = PanelConfig(
        schema_version=1, panel_version=1, panel=[PanelEntry("greedy", "greedy")],
        match=PanelMatchOptions(episodes=2, agent_player="random", device="cpu"),
        history=[],
    )
    entry_a = _stub_entry(a_ckpt, "a_run_latest", cfg)
    entry_b = _stub_entry(b_ckpt, "b_run_latest", cfg)

    new_a, new_b = run_pairwise(entry_a, entry_b, panel, episodes=2)

    assert len(new_a.pairwise_results) == 1
    assert len(new_b.pairwise_results) == 1
    a_rec = new_a.pairwise_results[0]
    b_rec = new_b.pairwise_results[0]
    assert a_rec.vs_entry_id == "b_run_latest"
    assert b_rec.vs_entry_id == "a_run_latest"
    assert a_rec.wins + b_rec.wins == a_rec.episodes
