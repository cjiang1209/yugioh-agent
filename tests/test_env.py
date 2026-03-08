"""Full environment integration tests."""

import pytest

from yugioh_env.deck_parser import parse_ydk
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
from yugioh_env.models import YuGiOhAction


@pytest.fixture
def env(lib, db_path, script_dirs, deck_path):
    """Create a YuGiOhEnvironment instance."""
    config = {
        "lib_path": None,  # auto-detect
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent_type": "random",
        "opponent_seed": 42,
    }
    e = YuGiOhEnvironment(config)
    yield e
    e.close()


def test_reset_returns_observation(env):
    """Reset should return a valid observation."""
    obs = env.reset(seed=42)
    assert obs is not None
    assert not obs.done
    assert len(obs.action_mask) == 32
    assert any(a == 1 for a in obs.action_mask)


def test_step_with_valid_action(env):
    """Step with a valid action should not crash."""
    obs = env.reset(seed=42)
    # Find first valid action
    for i, mask in enumerate(obs.action_mask):
        if mask == 1:
            obs2 = env.step(YuGiOhAction(action_index=i))
            assert obs2 is not None
            break


def test_full_episode(env):
    """Play a full episode until done."""
    obs = env.reset(seed=42)
    steps = 0
    max_steps = 500

    while not obs.done and steps < max_steps:
        # Pick first valid action
        action_idx = 0
        for i, mask in enumerate(obs.action_mask):
            if mask == 1:
                action_idx = i
                break
        obs = env.step(YuGiOhAction(action_index=action_idx))
        steps += 1

    # Game should have ended
    if obs.done:
        assert obs.reward in (1.0, -1.0, 0.0)


def test_state_property(env):
    """State property should return valid YuGiOhState."""
    env.reset(seed=42)
    state = env.state
    assert state.my_lp == 8000
    assert state.opp_lp == 8000
    assert state.turn_count >= 0


def test_multiple_episodes(env):
    """Should be able to play multiple episodes."""
    for seed in range(3):
        obs = env.reset(seed=seed)
        steps = 0
        while not obs.done and steps < 200:
            for i, mask in enumerate(obs.action_mask):
                if mask == 1:
                    obs = env.step(YuGiOhAction(action_index=i))
                    break
            steps += 1


# --- Deck-at-reset tests ---


@pytest.fixture
def inline_deck(deck_path):
    """Parse the starter deck file into an inline dict."""
    return parse_ydk(deck_path)


def test_reset_with_inline_decks(env, inline_deck):
    """Reset with inline deck dicts for both players."""
    obs = env.reset(seed=100, deck0=inline_deck, deck1=inline_deck)
    assert obs is not None
    assert not obs.done
    assert any(a == 1 for a in obs.action_mask)


def test_reset_with_one_inline_deck(env, inline_deck):
    """Reset with one inline deck; other uses server default."""
    obs = env.reset(seed=101, deck0=inline_deck)
    assert obs is not None
    assert not obs.done
    assert any(a == 1 for a in obs.action_mask)


def test_reset_deck_validation_rejects_empty_main(env):
    """Empty main deck should raise ValueError."""
    bad_deck = {"main": [], "extra": []}
    with pytest.raises(ValueError, match="40-60 cards"):
        env.reset(seed=200, deck0=bad_deck)


def test_reset_deck_validation_rejects_bad_codes(env):
    """Negative card codes should raise ValueError."""
    bad_deck = {"main": [-1] * 40}
    with pytest.raises(ValueError, match="positive integers"):
        env.reset(seed=201, deck0=bad_deck)


def test_reset_deck_validation_rejects_oversized_extra(env):
    """>15 extra deck cards should raise ValueError."""
    bad_deck = {"main": [89631139] * 40, "extra": [89631139] * 16}
    with pytest.raises(ValueError, match="0-15 cards"):
        env.reset(seed=202, deck0=bad_deck)
