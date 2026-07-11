import sys


def test_deck_allocation_flag_reaches_config(tmp_path, monkeypatch):
    from cli.train import _build_fresh_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.train",
            "--deck-paths",
            "assets/decks/ygo_agent_blueeyes.ydk",
            "--deck-allocation",
            "balanced",
            "--mirror-decks",
            "--total-timesteps",
            "0",
        ],
    )
    cfg = _build_fresh_config(parse_args(), str(tmp_path / "run"))
    assert cfg.deck_allocation == "balanced"
    assert cfg.mirror_decks is True


def test_deck_selection_flag_defaults(tmp_path, monkeypatch):
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
    cfg = _build_fresh_config(parse_args(), str(tmp_path / "run"))
    assert cfg.deck_allocation == "random"
    assert cfg.mirror_decks is False
