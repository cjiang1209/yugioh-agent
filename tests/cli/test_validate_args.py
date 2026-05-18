"""Tests for cli.train.validate_args — errors for conflicting arguments,
warnings when one argument voids another."""

from __future__ import annotations

import subprocess
import sys


# ---------------------------------------------------------------------------
# Mutual-exclusivity errors (hard failures)
# ---------------------------------------------------------------------------

def test_resume_and_init_mutually_exclusive():
    """CLI should error when both --resume and --init-checkpoint are given."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--resume", "fake.pt",
            "--init-checkpoint", "fake2.pt",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "--resume and --init-checkpoint are mutually exclusive" in result.stderr


def test_resume_with_init_optimizer_rejected():
    """--init-optimizer is for --init-checkpoint, not --resume."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--resume", "fake.pt",
            "--init-optimizer",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "--init-optimizer is for use with --init-checkpoint" in result.stderr


def test_init_optimizer_without_init_checkpoint():
    """--init-optimizer without --init-checkpoint should exit with error."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train", "--init-optimizer"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "requires --init-checkpoint" in result.stderr


# ---------------------------------------------------------------------------
# Eval-opponents validation
# ---------------------------------------------------------------------------

def test_eval_opponent_model_requires_path():
    """--eval-opponents model: must include a checkpoint path."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train", "--eval-opponents", "model:"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "must include a checkpoint path" in result.stderr


def test_eval_opponent_unknown_rejected():
    """Unknown eval opponent type should be rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train", "--eval-opponents", "bogus"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "unknown opponent" in result.stderr


def test_eval_opponent_model_checkpoint_not_found():
    """Nonexistent eval opponent checkpoint should be rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train",
         "--eval-opponents", "model:/nonexistent/checkpoint.pt"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr


# ---------------------------------------------------------------------------
# Voided-argument warnings
# ---------------------------------------------------------------------------

def test_resume_with_base_dir_warns(tmp_path):
    """--base-dir should produce a warning when --resume is used."""
    # Create a minimal file so the existence check passes; the subprocess
    # handles actual torch loading.
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.write_bytes(b"")

    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--resume", str(ckpt_path),
            "--base-dir", "/tmp/ignored",
            "--total-timesteps", "0",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert "--base-dir has no effect with --resume" in result.stderr


def test_shaping_lp_weight_ignored_without_shaping(tmp_path):
    """--shaping-lp-weight should warn when --no-reward-shaping is set."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--no-reward-shaping",
            "--shaping-lp-weight", "0.05",
            "--total-timesteps", "0",
            "--base-dir", str(tmp_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert "--shaping-lp-weight has no effect with --no-reward-shaping" in result.stderr


def test_shaping_card_weight_ignored_without_shaping(tmp_path):
    """--shaping-card-weight should warn when --no-reward-shaping is set."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--no-reward-shaping",
            "--shaping-card-weight", "0.05",
            "--total-timesteps", "0",
            "--base-dir", str(tmp_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert "--shaping-card-weight has no effect with --no-reward-shaping" in result.stderr


def test_no_reward_shaping_alone_no_warning(tmp_path):
    """--no-reward-shaping alone should NOT warn about shaping weights."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--no-reward-shaping",
            "--total-timesteps", "0",
            "--base-dir", str(tmp_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert "shaping-lp-weight has no effect" not in result.stderr
    assert "shaping-card-weight has no effect" not in result.stderr


def test_opponent_model_requires_path():
    """--opponent model: must include a checkpoint path."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train", "--opponent", "model:"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "must include a checkpoint path" in result.stderr


def test_opponent_unknown_rejected():
    """Unknown opponent type should be rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train", "--opponent", "bogus"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "unknown opponent" in result.stderr


def test_opponent_model_checkpoint_not_found():
    """Nonexistent opponent checkpoint should be rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.train",
         "--opponent", "model:/nonexistent/checkpoint.pt"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr


def test_self_play_rejects_sync_actor_learner():
    """--self-play with sync_actor_learner vec-env should be rejected."""
    result = subprocess.run(
        [
            sys.executable, "-m", "cli.train",
            "--self-play",
            "--vec-env-type", "sync_actor_learner",
            "--total-timesteps", "100",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "--self-play with --vec-env-type sync_actor_learner is not yet wired" in result.stderr
