"""Subprocess tests for cli.eval — argument validation and error messages.

Mirrors tests/cli/test_validate_args.py. Asserts that the same bad specs
produce the same error strings in both CLIs (so users see consistent feedback
across train and eval).
"""

from __future__ import annotations

import subprocess
import sys


def _run_eval(deck: str, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cli.eval", "--deck-paths", deck, *extra_args],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# --agent validation
# ---------------------------------------------------------------------------


def test_agent_model_requires_path(deck_path_str):
    result = _run_eval(deck_path_str, "--agent", "model:", "--opponents", "greedy")
    assert result.returncode != 0
    assert "must include a checkpoint path" in result.stderr


def test_agent_unknown_rejected(deck_path_str):
    result = _run_eval(deck_path_str, "--agent", "bogus", "--opponents", "greedy")
    assert result.returncode != 0
    assert "unknown opponent" in result.stderr


def test_agent_model_checkpoint_not_found(deck_path_str):
    result = _run_eval(
        deck_path_str,
        "--agent",
        "model:/nonexistent/checkpoint.pt",
        "--opponents",
        "greedy",
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr


# ---------------------------------------------------------------------------
# --opponents validation
# ---------------------------------------------------------------------------


def test_opponents_model_requires_path(deck_path_str):
    result = _run_eval(deck_path_str, "--agent", "greedy", "--opponents", "model:")
    assert result.returncode != 0
    assert "must include a checkpoint path" in result.stderr


def test_opponents_unknown_rejected(deck_path_str):
    result = _run_eval(deck_path_str, "--agent", "greedy", "--opponents", "bogus")
    assert result.returncode != 0
    assert "unknown opponent" in result.stderr


def test_opponents_model_checkpoint_not_found(deck_path_str):
    result = _run_eval(
        deck_path_str,
        "--agent",
        "greedy",
        "--opponents",
        "model:/nonexistent/checkpoint.pt",
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr


def test_opponents_one_bad_in_list_rejected(deck_path_str):
    """Mixed valid+invalid --opponents list should fail on the bad entry."""
    result = _run_eval(
        deck_path_str,
        "--agent",
        "greedy",
        "--opponents",
        "greedy",
        "model:",
        "random",
    )
    assert result.returncode != 0
    assert "must include a checkpoint path" in result.stderr


# ---------------------------------------------------------------------------
# Required-arg checks (argparse layer, not our validators)
# ---------------------------------------------------------------------------


def test_missing_agent_rejected(deck_path_str):
    result = subprocess.run(
        [sys.executable, "-m", "cli.eval", "--deck-paths", deck_path_str, "--opponents", "greedy"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--agent" in result.stderr


def test_missing_opponents_rejected(deck_path_str):
    result = subprocess.run(
        [sys.executable, "-m", "cli.eval", "--deck-paths", deck_path_str, "--agent", "greedy"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--opponents" in result.stderr


def test_missing_deck_paths_rejected():
    result = subprocess.run(
        [sys.executable, "-m", "cli.eval", "--agent", "greedy", "--opponents", "greedy"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--deck-paths" in result.stderr


def test_deck_path_not_found():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.eval",
            "--deck-paths",
            "/nope.ydk",
            "--agent",
            "greedy",
            "--opponents",
            "greedy",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "deck file not found" in result.stderr


def test_negative_episodes_rejected(deck_path_str):
    result = _run_eval(
        deck_path_str,
        "--agent",
        "greedy",
        "--opponents",
        "greedy",
        "--episodes",
        "-1",
    )
    assert result.returncode != 0
    assert "must be >= 0" in result.stderr
