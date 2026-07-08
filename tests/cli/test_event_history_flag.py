import sys

from yugioh_rl.config import TrainingConfig


def test_event_history_flag_reaches_config(tmp_path, monkeypatch):
    from cli.train import _build_fresh_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--deck-paths",
            "assets/decks/ygo_agent_blueeyes.ydk",
            "--event-history-dim",
            "32",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))
    assert config.event_history_dim == 32


def test_event_history_defaults_zero(tmp_path, monkeypatch):
    from cli.train import _build_fresh_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--deck-paths",
            "assets/decks/ygo_agent_blueeyes.ydk",
            "--total-timesteps",
            "0",
        ],
    )
    args = parse_args()
    config = _build_fresh_config(args, str(tmp_path / "run"))
    assert config.event_history_dim == 0
    assert config.event_history_dim == TrainingConfig().event_history_dim
