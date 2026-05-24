"""Tests for entry file I/O and entry_id derivation."""

from __future__ import annotations

import json

import pytest

from yugioh_leaderboard.entry import (
    Entry,
    PairwiseMatchResult,
    PanelMatchResult,
    compute_checkpoint_hash,
    entry_id_for,
    read_entry,
    write_entry,
)


def _sample_entry() -> Entry:
    return Entry(
        schema_version=1,
        entry_id="20260411_143000_seed42_latest",
        checkpoint_path="checkpoints/run/checkpoint_latest.pt",
        checkpoint_hash="sha256:abc123",
        added_at="2026-04-11T14:30:00Z",
        panel_version=1,
        features={"rnn_type": "lstm", "seed": 42},
        tags=["v2"],
        panel_results=[
            PanelMatchResult(
                opponent_label="random",
                episodes=100,
                wins=95,
                win_rate=0.95,
                per_deck={"blue_eyes": {"episodes": 100, "wins": 95, "win_rate": 0.95}},
                seed=1000,
                evaluated_at="2026-04-11T14:35:00Z",
            )
        ],
        pairwise_results=[],
    )


def test_round_trip(tmp_path):
    e = _sample_entry()
    path = tmp_path / f"{e.entry_id}.json"
    write_entry(path, e)
    loaded = read_entry(path)
    assert loaded == e


def test_round_trip_through_json_bytes(tmp_path):
    e = _sample_entry()
    path = tmp_path / f"{e.entry_id}.json"
    write_entry(path, e)
    raw = json.loads(path.read_text())
    assert raw["entry_id"] == e.entry_id
    assert raw["panel_results"][0]["opponent_label"] == "random"


def test_entry_id_for_latest_symlink(tmp_path):
    run_dir = tmp_path / "20260411_143000_seed42"
    run_dir.mkdir()
    real = run_dir / "checkpoint_500.pt"
    real.write_bytes(b"fake")
    symlink = run_dir / "checkpoint_latest.pt"
    symlink.symlink_to("checkpoint_500.pt")

    assert entry_id_for(symlink) == "20260411_143000_seed42_latest"


def test_entry_id_for_numbered_checkpoint(tmp_path):
    run_dir = tmp_path / "20260411_143000_seed42"
    run_dir.mkdir()
    real = run_dir / "checkpoint_500.pt"
    real.write_bytes(b"fake")

    assert entry_id_for(real) == "20260411_143000_seed42_500"


def test_atomic_write_no_tmp_left_after_success(tmp_path):
    e = _sample_entry()
    path = tmp_path / f"{e.entry_id}.json"
    write_entry(path, e)
    assert path.exists()
    assert not (tmp_path / f"{e.entry_id}.json.tmp").exists()


def test_atomic_write_overwrites_existing(tmp_path):
    e = _sample_entry()
    path = tmp_path / f"{e.entry_id}.json"
    write_entry(path, e)
    e2 = _sample_entry()
    e2.tags = ["different"]
    write_entry(path, e2)
    assert read_entry(path).tags == ["different"]


def test_compute_checkpoint_hash_deterministic(tmp_path):
    p = tmp_path / "ckpt.pt"
    p.write_bytes(b"hello world")
    h1 = compute_checkpoint_hash(p)
    h2 = compute_checkpoint_hash(p)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert h1 == "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_compute_checkpoint_hash_changes_with_content(tmp_path):
    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    p1.write_bytes(b"alpha")
    p2.write_bytes(b"beta")
    assert compute_checkpoint_hash(p1) != compute_checkpoint_hash(p2)


def test_entry_id_for_rejects_unknown_pattern():
    with pytest.raises(ValueError, match="checkpoint_"):
        entry_id_for("/some/dir/model.pt")


def test_entry_id_for_rejects_path_without_run_dir():
    with pytest.raises(ValueError, match="run-directory parent"):
        entry_id_for("checkpoint_500.pt")


def test_panel_match_result_ignores_unknown_keys(tmp_path):
    e = _sample_entry()
    path = tmp_path / "e.json"
    write_entry(path, e)
    raw = json.loads(path.read_text())
    raw["panel_results"][0]["unknown_future_field"] = "ok"
    path.write_text(json.dumps(raw))
    loaded = read_entry(path)
    assert loaded.panel_results[0].opponent_label == "random"


def test_pairwise_match_result_ignores_unknown_keys(tmp_path):
    e = _sample_entry()
    e.pairwise_results = [
        PairwiseMatchResult(
            vs_entry_id="other",
            vs_checkpoint_hash="sha256:zzz",
            episodes=10,
            wins=5,
            win_rate=0.5,
            per_deck={},
            seed=1,
            evaluated_at="t",
        )
    ]
    path = tmp_path / "e.json"
    write_entry(path, e)
    raw = json.loads(path.read_text())
    raw["pairwise_results"][0]["future_field"] = 42
    path.write_text(json.dumps(raw))
    loaded = read_entry(path)
    assert loaded.pairwise_results[0].vs_entry_id == "other"
