"""Tests for pure Elo math (yugioh_rl.elo)."""
from __future__ import annotations

import pytest

from yugioh_rl.elo import expected_score, update


def test_expected_score_equal_ratings_is_half() -> None:
    assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)


def test_expected_score_400_point_gap_is_ten_to_one() -> None:
    # R_a is 400 above R_b -> expected score ~10/11 = 0.9090909...
    assert expected_score(1900.0, 1500.0) == pytest.approx(10.0 / 11.0, rel=1e-6)


def test_expected_score_negative_gap_is_one_minus() -> None:
    e_ab = expected_score(1700.0, 1500.0)
    e_ba = expected_score(1500.0, 1700.0)
    assert e_ab + e_ba == pytest.approx(1.0)


def test_update_winner_gains_what_loser_loses() -> None:
    # Zero-sum invariant.
    new_a, new_b = update(1500.0, 1500.0, agent_won=True, k=16.0)
    delta_a = new_a - 1500.0
    delta_b = 1500.0 - new_b
    assert delta_a == pytest.approx(delta_b)


def test_update_equal_ratings_win_moves_by_half_k() -> None:
    new_a, new_b = update(1500.0, 1500.0, agent_won=True, k=16.0)
    # S - E = 1 - 0.5 = 0.5, so delta = 0.5 * K = 8.0
    assert new_a == pytest.approx(1508.0)
    assert new_b == pytest.approx(1492.0)


def test_update_loss_is_negative() -> None:
    new_a, new_b = update(1500.0, 1500.0, agent_won=False, k=16.0)
    assert new_a == pytest.approx(1492.0)
    assert new_b == pytest.approx(1508.0)


def test_update_large_gap_winner_gains_little() -> None:
    # Heavy favorite winning barely moves the needle.
    new_a, _ = update(1900.0, 1500.0, agent_won=True, k=16.0)
    delta = new_a - 1900.0
    assert 0 < delta < 2.0  # 16 * (1 - 10/11) ~= 1.45


def test_update_upset_is_large() -> None:
    # Underdog winning gains a lot.
    new_a, _ = update(1500.0, 1900.0, agent_won=True, k=16.0)
    delta = new_a - 1500.0
    assert delta > 14.0  # 16 * (1 - 1/11) ~= 14.55
