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
