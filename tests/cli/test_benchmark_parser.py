"""Tests for cli.benchmark_throughput.parse_log late-window selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.benchmark_throughput import parse_log


def _write_log(path: Path, rows: list[tuple[int, int, int]]) -> None:
    """rows is [(update_idx, total_steps, seconds_offset), ...]."""
    lines = []
    for upd, steps, sec in rows:
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        lines.append(
            f"{h:02d}:{m:02d}:{s:02d} [INFO] yugioh_rl.ppo: "
            f"Update {upd}/100 | steps={steps} | FPS=999"
        )
    path.write_text("\n".join(lines) + "\n")


def _rows(n: int, *, steps_per_update: int = 1024, seconds_per_update: int = 1):
    return [
        (i + 1, (i + 1) * steps_per_update, (i + 1) * seconds_per_update)
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 10, 15, 25, 35])
def test_parse_log_handles_all_row_counts(tmp_path: Path, n: int) -> None:
    log = tmp_path / f"n{n}.log"
    _write_log(log, _rows(n))
    out = parse_log(log)
    assert out is not None, f"parse_log returned None for n={n}"
    assert out["updates_logged"] == n
    assert out["final_update"] == n
    # 1024 steps per 1 second → 1024 FPS.
    assert out["steady_fps"] == pytest.approx(1024.0)


def test_parse_log_skips_warmup_row(tmp_path: Path) -> None:
    """First row's wall-time is contaminated; the window must exclude it."""
    log = tmp_path / "warmup.log"
    rows = [
        (1, 1024, 100),   # update 1: very slow due to warmup (100s gap from t=0)
        (2, 2048, 101),   # steady: 1024 steps / 1s = 1024 FPS
        (3, 3072, 102),
        (4, 4096, 103),
        (5, 5120, 104),
    ]
    _write_log(log, rows)
    out = parse_log(log)
    assert out is not None
    assert out["steady_fps"] == pytest.approx(1024.0), \
        "warmup row should be excluded; expected steady-state FPS"


def test_parse_log_returns_none_for_empty(tmp_path: Path) -> None:
    log = tmp_path / "empty.log"
    log.write_text("")
    assert parse_log(log) is None


def test_parse_log_returns_none_for_single_row(tmp_path: Path) -> None:
    log = tmp_path / "single.log"
    _write_log(log, _rows(1))
    assert parse_log(log) is None


def test_parse_log_handles_midnight_wrap(tmp_path: Path) -> None:
    """Timestamps wrap at midnight; selector must compute positive dt."""
    log = tmp_path / "midnight.log"
    rows = [
        (1, 1024, 86399),  # 23:59:59
        (2, 2048, 0),       # 00:00:00 next day
        (3, 3072, 1),       # 00:00:01
    ]
    _write_log(log, rows)
    out = parse_log(log)
    assert out is not None
    # dt should be (1 - 0) % 86400 = 1 second across rows[1] -> rows[2].
    assert out["steady_fps"] == pytest.approx(1024.0)
