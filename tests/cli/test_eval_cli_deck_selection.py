def test_eval_cli_defaults_random():
    from cli.eval import parse_args

    a = parse_args(
        [
            "--agent",
            "random",
            "--opponents",
            "greedy",
            "--deck-paths",
            "assets/decks/ygo_agent_blueeyes.ydk",
        ]
    )
    assert a.deck_allocation == "random"
    assert a.mirror_decks is False


def test_eval_cli_overrides():
    from cli.eval import parse_args

    a = parse_args(
        [
            "--agent",
            "random",
            "--opponents",
            "greedy",
            "--deck-paths",
            "assets/decks/ygo_agent_blueeyes.ydk",
            "--deck-allocation",
            "balanced",
            "--mirror-decks",
        ]
    )
    assert a.deck_allocation == "balanced"
    assert a.mirror_decks is True


def test_sweep_cli_defaults_random():
    from cli.eval_sweep import parse_args

    a = parse_args(["--run-dir", "r", "--opponents", "random"])
    assert a.deck_allocation == "random"
    assert a.mirror_decks is False
