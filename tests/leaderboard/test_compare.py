"""Tests for bootstrap statistics + group comparison logic."""

from __future__ import annotations

import pytest

from yugioh_leaderboard.compare import (
    ComparisonResult,
    bootstrap_ci,
    compare_groups,
    format_comparison_table,
    paired_bootstrap_delta,
)


def test_constant_input_produces_collapsed_ci():
    lo, hi = bootstrap_ci([0.5] * 5)
    assert lo == 0.5
    assert hi == 0.5


def test_constant_zero_input_produces_zero_ci():
    lo, hi = bootstrap_ci([0.0] * 5)
    assert lo == 0.0
    assert hi == 0.0


def test_constant_one_input_produces_one_ci():
    lo, hi = bootstrap_ci([1.0] * 5)
    assert lo == 1.0
    assert hi == 1.0


def test_mixed_input_ci_brackets_mean():
    values = [0.3, 0.5, 0.7]
    lo, hi = bootstrap_ci(values)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_is_deterministic():
    a = bootstrap_ci([0.4, 0.5, 0.6, 0.7])
    b = bootstrap_ci([0.4, 0.5, 0.6, 0.7])
    assert a == b


def test_n_eq_1_returns_point_estimate():
    lo, hi = bootstrap_ci([0.42])
    assert lo == 0.42
    assert hi == 0.42


def test_paired_delta_zero_when_groups_identical():
    a = {42: 0.5, 43: 0.6, 44: 0.7}
    delta, lo, hi = paired_bootstrap_delta(a, a)
    assert delta == 0.0
    assert lo <= 0 <= hi


def test_paired_delta_positive_when_b_consistently_higher():
    a = {42: 0.5, 43: 0.6, 44: 0.7}
    b = {42: 0.6, 43: 0.7, 44: 0.8}
    delta, lo, hi = paired_bootstrap_delta(a, b)
    assert abs(delta - 0.1) < 1e-9
    assert lo > 0


def test_paired_delta_uses_intersection_of_seeds():
    a = {42: 0.5, 43: 0.5, 99: 0.0}
    b = {42: 0.6, 43: 0.6, 100: 1.0}
    delta, lo, hi = paired_bootstrap_delta(a, b)
    assert abs(delta - 0.1) < 1e-9


def test_paired_delta_raises_on_empty_intersection():
    a = {42: 0.5}
    b = {99: 0.5}
    with pytest.raises(ValueError, match="no shared seeds"):
        paired_bootstrap_delta(a, b)


def test_compare_groups_by_rnn_type_paired(make_entry):
    entries = [
        make_entry("a1", "none", 42, 0.5, 0.4),
        make_entry("a2", "none", 43, 0.5, 0.4),
        make_entry("a3", "none", 44, 0.5, 0.4),
        make_entry("b1", "lstm", 42, 0.6, 0.5),
        make_entry("b2", "lstm", 43, 0.6, 0.5),
        make_entry("b3", "lstm", 44, 0.6, 0.5),
    ]
    result = compare_groups(entries, by_field="rnn_type")
    assert isinstance(result, ComparisonResult)
    assert set(result.groups.keys()) == {"none", "lstm"}
    assert result.baseline_group == "lstm"  # alphabetically first
    assert result.groups["none"].n_seeds == 3
    assert result.groups["lstm"].n_seeds == 3
    delta_none = result.paired_delta_by_group["none"]["greedy"]
    assert abs(delta_none[0] - (-0.1)) < 1e-9


def test_compare_groups_skips_when_only_one_value(make_entry):
    entries = [make_entry("a1", "lstm", 42, 0.5, 0.5)]
    result = compare_groups(entries, by_field="rnn_type")
    assert result.skip_reason == "only 1 value of `rnn_type`"


def test_compare_groups_filter_excludes_entries(make_entry):
    entries = [
        make_entry("a1", "none", 42, 0.5, 0.4),
        make_entry("b1", "lstm", 42, 0.6, 0.5),
    ]
    result = compare_groups(
        entries, by_field="rnn_type", filter={"rnn_type": "lstm"}
    )
    assert result.skip_reason is not None


def test_format_comparison_table_shows_paired_delta(make_entry):
    entries = [
        make_entry("a1", "none", 42, 0.5, 0.4),
        make_entry("a2", "none", 43, 0.5, 0.4),
        make_entry("a3", "none", 44, 0.5, 0.4),
        make_entry("b1", "lstm", 42, 0.6, 0.5),
        make_entry("b2", "lstm", 43, 0.6, 0.5),
        make_entry("b3", "lstm", 44, 0.6, 0.5),
    ]
    result = compare_groups(entries, by_field="rnn_type")
    text = format_comparison_table(result)
    assert "rnn_type" in text
    assert "none" in text
    assert "lstm" in text
    assert "vs random" in text
    assert "vs greedy" in text
    assert "-0.10" in text  # none is non-baseline; delta = none - lstm = -0.10


def test_format_table_renders_ci_as_bracket_pair(make_entry):
    """Win-rate cells show [lo, hi] so asymmetric CIs near 0/1 stay visible."""
    entries = [
        make_entry("a1", "none", 42, 0.5, 0.4),
        make_entry("a2", "none", 43, 0.5, 0.4),
        make_entry("b1", "lstm", 42, 0.6, 0.5),
        make_entry("b2", "lstm", 43, 0.6, 0.5),
    ]
    result = compare_groups(entries, by_field="rnn_type")
    text = format_comparison_table(result)
    assert "[0.50, 0.50]" in text
    assert "[0.60, 0.60]" in text
    assert " ± " not in text


def test_compare_filter_matches_bool_feature(make_entry):
    """`reward_shaping=true` (string from CLI) must match Python bool True."""
    entries = [
        make_entry("a", "lstm", 42, 0.6, 0.5),
        make_entry("b", "lstm", 43, 0.6, 0.5),
        make_entry("c", "none", 42, 0.5, 0.4),
        make_entry("d", "none", 43, 0.5, 0.4),
    ]
    result = compare_groups(entries, by_field="rnn_type", filter={"reward_shaping": "true"})
    assert result.skip_reason is None
    assert set(result.groups.keys()) == {"lstm", "none"}


def test_compare_filter_excluded_all_returns_filter_specific_reason(make_entry):
    entries = [make_entry("a", "lstm", 42, 0.6, 0.5)]
    result = compare_groups(
        entries, by_field="rnn_type", filter={"rnn_type": "lstm"}
    )
    assert result.skip_reason is not None
    assert "filter" in result.skip_reason
