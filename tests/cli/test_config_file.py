"""Tests for cli.train --config JSON loading and merge precedence."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli.train import _load_config_file

from yugioh_rl.config import TrainingConfig


def _write_config(path: Path, fields: dict) -> Path:
    path.write_text(json.dumps(fields))
    return path


def test_config_with_resume_rejected(tmp_path):
    """--config and --resume are mutually exclusive.

    The mutex fires before --resume's path-existence check, so we don't need
    to materialize ckpt.pt — a fake path is sufficient.
    """
    cfg = _write_config(tmp_path / "c.json", {"learning_rate": 1e-4})

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.train",
            "--config",
            str(cfg),
            "--resume",
            "fake.pt",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--config and --resume are mutually exclusive" in result.stderr


def test_unknown_json_key_fatal(tmp_path, capsys):
    """Unknown JSON keys cause a fatal error listing them."""
    cfg = _write_config(tmp_path / "c.json", {"bogus_field": 1, "rnn_type": "lstm"})

    with pytest.raises(SystemExit) as exc:
        _load_config_file(cfg)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown keys" in err
    assert "bogus_field" in err


def test_config_file_not_json_object(tmp_path):
    """A JSON file whose top level is not an object is rejected."""
    cfg = tmp_path / "c.json"
    cfg.write_text("[1, 2, 3]")

    with pytest.raises(SystemExit) as exc:
        _load_config_file(cfg)
    assert exc.value.code == 2


def test_config_file_invalid_json(tmp_path):
    """A malformed JSON file is rejected with a parser error."""
    cfg = tmp_path / "c.json"
    cfg.write_text("{not json")

    with pytest.raises(SystemExit) as exc:
        _load_config_file(cfg)
    assert exc.value.code == 2


def test_config_file_not_found(tmp_path):
    """A missing --config path is rejected."""
    with pytest.raises(SystemExit) as exc:
        _load_config_file(tmp_path / "missing.json")
    assert exc.value.code == 2


def test_config_loader_drops_derived_fields(tmp_path):
    """save_dir and resume_checkpoint are silently dropped from JSON;
    init_checkpoint and init_optimizer pass through (they're configured)."""
    cfg = _write_config(
        tmp_path / "c.json",
        {
            "save_dir": "/should/be/dropped",
            "resume_checkpoint": "/also/dropped.pt",
            "init_checkpoint": "/keep/this.pt",
            "init_optimizer": True,
            "rnn_type": "gru",
        },
    )
    result = _load_config_file(cfg)
    assert "save_dir" not in result
    assert "resume_checkpoint" not in result
    assert result == {
        "init_checkpoint": "/keep/this.pt",
        "init_optimizer": True,
        "rnn_type": "gru",
    }


def test_partial_config_uses_defaults_for_missing(tmp_path, monkeypatch):
    """A JSON with one key sets that field; all others stay at defaults."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(tmp_path / "c.json", {"rnn_type": "lstm"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    defaults = TrainingConfig()
    assert config.rnn_type == "lstm"
    assert config.learning_rate == defaults.learning_rate
    assert config.num_envs == defaults.num_envs
    assert config.gamma == defaults.gamma


def test_json_value_used_when_no_cli_flag(tmp_path, monkeypatch):
    """JSON values flow through when the CLI doesn't override them."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(
        tmp_path / "c.json",
        {
            "rnn_type": "gru",
            "learning_rate": 1.5e-4,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.rnn_type == "gru"
    assert config.learning_rate == 1.5e-4


def test_cli_flag_overrides_json(tmp_path, monkeypatch):
    """An explicitly passed CLI flag wins over the same key in JSON."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(tmp_path / "c.json", {"learning_rate": 1e-4})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--learning-rate",
            "5e-4",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.learning_rate == 5e-4


def test_no_reward_shaping_flag_inverts_json(tmp_path, monkeypatch):
    """--no-reward-shaping CLI flag wins over `reward_shaping: true` in JSON."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(tmp_path / "c.json", {"reward_shaping": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--no-reward-shaping",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.reward_shaping is False


def test_card_embeddings_cli_flag_overrides_json(tmp_path, monkeypatch):
    """The --card-embeddings CLI flag overrides the same field in JSON."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(tmp_path / "c.json", {"card_embeddings": "from_json.pt"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--card-embeddings",
            "from_cli.pt",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.card_embeddings == "from_cli.pt"


def test_derived_fields_dropped_from_json(tmp_path, monkeypatch):
    """`save_dir` and `resume_checkpoint` in JSON don't override CLI-derived values."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(
        tmp_path / "c.json",
        {
            "save_dir": "/should/be/ignored",
            "resume_checkpoint": "/also/ignored.pt",
            "rnn_type": "gru",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    save_dir = str(tmp_path / "run")
    config = _build_fresh_config(args, save_dir)

    assert config.save_dir == save_dir
    assert config.resume_checkpoint == ""
    assert config.rnn_type == "gru"


def test_complete_snapshot_loads(tmp_path, monkeypatch):
    """A full TrainingConfig serialized via asdict() loads back losslessly
    after the derived fields are dropped."""
    from cli.train import _build_fresh_config, parse_args

    original = TrainingConfig(
        rnn_type="lstm",
        rnn_hidden_dim=128,
        learning_rate=2.5e-4,
        save_dir="/old/run/dir",  # derived; will be dropped
    )
    cfg = _write_config(tmp_path / "c.json", dataclasses.asdict(original))

    monkeypatch.setattr(sys, "argv", ["cli.train", "--config", str(cfg)])
    args = parse_args()
    save_dir = str(tmp_path / "new_run")
    config = _build_fresh_config(args, save_dir)

    assert config.rnn_type == "lstm"
    assert config.rnn_hidden_dim == 128
    assert config.learning_rate == 2.5e-4
    assert config.save_dir == save_dir


def test_config_with_init_checkpoint_allowed(tmp_path, monkeypatch):
    """--config + --init-checkpoint is permitted (both honored)."""
    from cli.train import _build_fresh_config, parse_args, validate_cli_args

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    cfg = _write_config(tmp_path / "c.json", {"learning_rate": 1e-4})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--init-checkpoint",
            str(ckpt),
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    validate_cli_args(args)  # must not fatal
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.init_checkpoint == str(ckpt)
    assert config.learning_rate == 1e-4


def test_list_field_cli_overrides_json_wholesale(tmp_path, monkeypatch):
    """A list-typed field set in JSON is replaced (not merged) by the CLI flag."""
    from cli.train import _build_fresh_config, parse_args

    cfg = _write_config(
        tmp_path / "c.json",
        {
            "eval_opponents": ["greedy", "random"],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--config",
            str(cfg),
            "--eval-opponents",
            "greedy",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))

    assert config.eval_opponents == ["greedy"]
