"""argparse + exit-code validation for the leaderboard CLI."""

from __future__ import annotations

import pytest

from cli.leaderboard import build_parser, _validate_subcommand_args


def test_parser_accepts_add_with_path():
    parser = build_parser()
    ns = parser.parse_args(["add", "/some/path.pt"])
    assert ns.command == "add"
    assert ns.checkpoint_path == "/some/path.pt"


def test_parser_accepts_add_tags_list():
    parser = build_parser()
    ns = parser.parse_args(["add", "/p.pt", "--tags", "v1", "ablation-3"])
    assert ns.tags == ["v1", "ablation-3"]


def test_parser_rejects_negative_episodes():
    parser = build_parser()
    ns = parser.parse_args(["add", "/p.pt", "--episodes", "-5"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_parser_rejects_missing_checkpoint(tmp_path):
    parser = build_parser()
    ns = parser.parse_args(["add", str(tmp_path / "nonexistent.pt")])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_refresh_index_subcommand_parses():
    parser = build_parser()
    ns = parser.parse_args(["refresh-index"])
    assert ns.command == "refresh-index"


def test_tags_default_none_distinct_from_clear_tags():
    parser = build_parser()
    no_tags = parser.parse_args(["add", "/p.pt"])
    assert no_tags.tags is None
    assert no_tags.clear_tags is False

    explicit = parser.parse_args(["add", "/p.pt", "--tags", "v1"])
    assert explicit.tags == ["v1"]

    cleared = parser.parse_args(["add", "/p.pt", "--clear-tags"])
    assert cleared.clear_tags is True


def test_compare_subcommand_parses_filter():
    parser = build_parser()
    ns = parser.parse_args([
        "compare", "--by", "rnn_type",
        "--filter", "reward_shaping=true", "deck_paths=a,b",
    ])
    assert ns.command == "compare"
    assert ns.by == "rnn_type"
    assert ns.filter == ["reward_shaping=true", "deck_paths=a,b"]


def test_compare_filter_bad_format_rejected():
    parser = build_parser()
    ns = parser.parse_args([
        "compare", "--by", "rnn_type",
        "--filter", "reward_shaping",
    ])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_compare_supports_include_stale_flag():
    parser = build_parser()
    ns = parser.parse_args(["compare", "--by", "rnn_type", "--include-stale"])
    assert ns.include_stale is True


def test_compare_by_unknown_field_rejected():
    parser = build_parser()
    ns = parser.parse_args(["compare", "--by", "definitely_not_a_field"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_compare_requires_by_or_by_tag():
    parser = build_parser()
    ns = parser.parse_args(["compare"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_pairwise_subcommand_parses():
    parser = build_parser()
    ns = parser.parse_args(["pairwise", "a_run_latest", "b_run_latest"])
    assert ns.command == "pairwise"
    assert ns.entry_a_id == "a_run_latest"
    assert ns.entry_b_id == "b_run_latest"
    assert ns.episodes == 100


def test_pairwise_rejects_negative_episodes():
    parser = build_parser()
    ns = parser.parse_args(["pairwise", "a", "b", "--episodes", "-1"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


# ---------------------------------------------------------------------------
# --workers plumbing on add and pairwise subcommands
# ---------------------------------------------------------------------------


def test_add_accepts_workers_flag():
    parser = build_parser()
    ns = parser.parse_args(["add", "/p.pt", "--workers", "4"])
    assert ns.workers == 4


def test_add_rejects_zero_workers(tmp_path):
    """--workers=0 must fail validation. Use a real path so the
    missing-checkpoint check doesn't fire first."""
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"")
    parser = build_parser()
    ns = parser.parse_args(["add", str(fake_ckpt), "--workers", "0"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


def test_pairwise_accepts_workers_flag():
    parser = build_parser()
    ns = parser.parse_args(["pairwise", "a", "b", "--workers", "2"])
    assert ns.workers == 2


def test_pairwise_rejects_zero_workers():
    parser = build_parser()
    ns = parser.parse_args(["pairwise", "a", "b", "--workers", "0"])
    with pytest.raises(SystemExit):
        _validate_subcommand_args(ns)


# ---------------------------------------------------------------------------
# End-to-end forwarding: --workers reaches score_checkpoint / run_pairwise
# ---------------------------------------------------------------------------


def test_add_forwards_workers_to_score_checkpoint(tmp_path, monkeypatch):
    """End-to-end: ``leaderboard add ... --workers 4`` invokes
    ``score_checkpoint(workers=4)``. Mocks the heavy bits."""
    from unittest.mock import patch
    from cli.leaderboard import _cmd_add
    from yugioh_leaderboard.entry import Entry

    fake_ckpt = tmp_path / "checkpoint_latest.pt"
    fake_ckpt.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_score_checkpoint(*args, **kwargs):
        captured.update(kwargs)
        return Entry(
            schema_version=1, entry_id="fake", checkpoint_path=str(fake_ckpt),
            checkpoint_hash="sha256:x", added_at="t",
            panel_version=1, features={}, tags=[],
            panel_results=[], pairwise_results=[],
        )

    parser = build_parser()
    ns = parser.parse_args(["add", str(fake_ckpt), "--workers", "4", "--force"])
    with patch("yugioh_leaderboard.score.score_checkpoint", fake_score_checkpoint), \
         patch("cli.leaderboard._load_panel"), \
         patch("yugioh_leaderboard.entry.compute_checkpoint_hash", return_value="sha256:x"), \
         patch("yugioh_leaderboard.entry.entry_id_for", return_value="fake"), \
         patch("yugioh_leaderboard.entry.write_entry"), \
         patch("cli.leaderboard._refresh_index_file"):
        _cmd_add(ns)

    assert captured["workers"] == 4


def test_pairwise_forwards_workers_to_run_pairwise(tmp_path, monkeypatch):
    from unittest.mock import patch
    from cli.leaderboard import _cmd_pairwise
    from yugioh_leaderboard.entry import Entry

    # Redirect the leaderboard's entries dir to tmp_path so we can drop in
    # fake entry files that pass the existence check in _cmd_pairwise.
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    (entries_dir / "a.json").write_text("{}")
    (entries_dir / "b.json").write_text("{}")
    fake_ckpt = tmp_path / "x.pt"
    fake_ckpt.write_bytes(b"")
    monkeypatch.setattr("cli.leaderboard.ENTRIES_DIR", entries_dir)

    def _stub_entry(eid: str) -> Entry:
        return Entry(
            schema_version=1, entry_id=eid, checkpoint_path=str(fake_ckpt),
            checkpoint_hash="sha256:x", added_at="t",
            panel_version=1, features={}, tags=[],
            panel_results=[], pairwise_results=[],
        )

    from yugioh_leaderboard.entry import PairwiseMatchResult

    captured: dict = {}

    def fake_run_pairwise(entry_a, entry_b, panel, **kwargs):
        captured.update(kwargs)
        # Populate pairwise_results so the CLI's post-call `next(...)` lookup
        # can resolve a record by vs_entry_id.
        rec_a = PairwiseMatchResult(
            vs_entry_id=entry_b.entry_id, vs_checkpoint_hash="sha256:x",
            episodes=1, wins=1, win_rate=1.0, per_deck={}, seed=0, evaluated_at="t",
        )
        rec_b = PairwiseMatchResult(
            vs_entry_id=entry_a.entry_id, vs_checkpoint_hash="sha256:x",
            episodes=1, wins=0, win_rate=0.0, per_deck={}, seed=0, evaluated_at="t",
        )
        new_a = entry_a.__class__(**{**entry_a.__dict__, "pairwise_results": [rec_a]})
        new_b = entry_b.__class__(**{**entry_b.__dict__, "pairwise_results": [rec_b]})
        return new_a, new_b

    def fake_read_entry(path):
        return _stub_entry(path.stem)

    parser = build_parser()
    ns = parser.parse_args(["pairwise", "a", "b", "--workers", "2"])
    with patch("yugioh_leaderboard.pairwise.run_pairwise", fake_run_pairwise), \
         patch("cli.leaderboard._load_panel"), \
         patch("yugioh_leaderboard.entry.read_entry", fake_read_entry), \
         patch("yugioh_leaderboard.entry.write_entry"), \
         patch("cli.leaderboard._refresh_index_file"):
        _cmd_pairwise(ns)

    assert captured["workers"] == 2
