"""Tests for multi-deck training support."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# parse_deck_pool
# ---------------------------------------------------------------------------


@pytest.fixture
def assets_dir():
    return Path(__file__).resolve().parents[2] / "assets"


def test_parse_deck_pool(assets_dir):
    """parse_deck_pool returns correct dicts for .ydk files."""
    from yugioh_rl.env_wrapper import parse_deck_pool

    deck = str(assets_dir / "decks" / "blue_eyes.ydk")
    pool = parse_deck_pool([deck])
    assert len(pool) == 1
    assert "main" in pool[0]
    assert len(pool[0]["main"]) >= 40
    assert all(isinstance(c, int) and c > 0 for c in pool[0]["main"])


def test_parse_deck_pool_multiple(assets_dir):
    """parse_deck_pool handles multiple deck files."""
    from yugioh_rl.env_wrapper import parse_deck_pool

    paths = [
        str(assets_dir / "decks" / "blue_eyes.ydk"),
        str(assets_dir / "decks" / "dark_magician.ydk"),
    ]
    pool = parse_deck_pool(paths)
    assert len(pool) == 2
    # Different decks should have different card lists
    assert pool[0]["main"] != pool[1]["main"]


def test_parse_deck_pool_picklable(assets_dir):
    """Pre-parsed deck dicts must be picklable for multiprocessing."""
    from yugioh_rl.env_wrapper import parse_deck_pool

    pool = parse_deck_pool([str(assets_dir / "decks" / "blue_eyes.ydk")])
    roundtripped = pickle.loads(pickle.dumps(pool))
    assert roundtripped == pool


# ---------------------------------------------------------------------------
# TrainingEnv deck sampling (mocked environment)
# ---------------------------------------------------------------------------


def _make_fake_obs():
    """Return a minimal YuGiOhObservation-like object."""
    obs = MagicMock()
    obs.cards = [0] * (200 * 42)
    obs.global_state = [0] * 20
    obs.actions = [0] * (32 * 28)
    obs.action_mask = [0] * 32
    obs.reward = 0.0
    obs.done = False
    return obs


POOL = [
    {"main": list(range(1, 41)), "extra": []},
    {"main": list(range(101, 141)), "extra": []},
    {"main": list(range(201, 241)), "extra": []},
]


def test_training_env_passes_deck_dicts():
    """reset() should pass deck0/deck1 dicts from the pool to env.reset()."""
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_env.reset.return_value = _make_fake_obs()
        mock_env._agent_player = 0

        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(deck_pool=POOL, seed=42)
        env.reset()

        reset_call = mock_env.reset.call_args
        assert "deck0" in reset_call.kwargs
        assert "deck1" in reset_call.kwargs
        assert reset_call.kwargs["deck0"] in POOL
        assert reset_call.kwargs["deck1"] in POOL


def test_training_env_agent_deck_idx_when_first():
    """When agent is player 0, deck0 should be the agent's deck."""
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_env.reset.return_value = _make_fake_obs()
        mock_env._agent_player = 0

        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(deck_pool=POOL, seed=42, agent_player="first")
        env.reset()

        reset_call = mock_env.reset.call_args
        # agent_player=0, so deck0 is the agent deck
        assert reset_call.kwargs["agent_player"] == 0
        agent_deck = reset_call.kwargs["deck0"]
        assert agent_deck == POOL[env._last_agent_deck_idx]


def test_training_env_agent_deck_idx_when_second():
    """When agent is player 1, deck1 should be the agent's deck."""
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_env.reset.return_value = _make_fake_obs()
        mock_env._agent_player = 1

        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(deck_pool=POOL, seed=42, agent_player="second")
        env.reset()

        reset_call = mock_env.reset.call_args
        assert reset_call.kwargs["agent_player"] == 1
        # deck1 is the agent's deck when agent_player=1
        agent_deck = reset_call.kwargs["deck1"]
        assert agent_deck == POOL[env._last_agent_deck_idx]


def test_training_env_deck_info_on_done():
    """On episode end, info should contain agent_deck_idx."""
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_env.reset.return_value = _make_fake_obs()
        mock_env._agent_player = 0

        done_obs = _make_fake_obs()
        done_obs.done = True
        done_obs.reward = 1.0
        mock_env.step.return_value = done_obs
        mock_env._step_count = 5

        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(deck_pool=POOL, seed=42)
        env.reset()
        _, _, _, info = env.step(0)

        assert "agent_deck_idx" in info
        assert 0 <= info["agent_deck_idx"] < len(POOL)


def test_training_env_samples_vary_across_resets():
    """Multiple resets should produce different deck combinations."""
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_env.reset.return_value = _make_fake_obs()
        mock_env._agent_player = 0

        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(deck_pool=POOL, seed=42)
        agent_indices = set()
        for _ in range(20):
            env.reset()
            agent_indices.add(env._last_agent_deck_idx)

        # With 3 decks and 20 resets, we should see multiple distinct decks
        assert len(agent_indices) > 1
