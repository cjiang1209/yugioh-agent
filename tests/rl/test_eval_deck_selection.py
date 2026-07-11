from yugioh_rl.eval import _make_eval_env_kwargs


def test_make_eval_env_kwargs_includes_selection_params():
    kw = _make_eval_env_kwargs(
        [{"main": [1]}],
        "random",
        seed=42,
        agent_player="random",
        opponent_device=None,
        deck_allocation="balanced",
        mirror_decks=True,
    )
    assert kw["deck_allocation"] == "balanced"
    assert kw["mirror_decks"] is True
    assert kw["opponent"] == "random"


def test_make_eval_env_kwargs_defaults_random():
    kw = _make_eval_env_kwargs(
        [{"main": [1]}],
        "random",
        seed=42,
        agent_player="random",
        opponent_device=None,
    )
    assert kw["deck_allocation"] == "random"
    assert kw["mirror_decks"] is False
