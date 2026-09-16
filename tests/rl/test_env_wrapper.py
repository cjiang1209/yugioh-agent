"""Unit tests for the ``TrainingEnv`` lifecycle changes.

After the auto-reset removal, ``step()`` returns terminal obs on done and
``reset()`` accepts an optional ``episode_idx`` so episodes are addressable
by index.  ``_deck_rng`` is reseeded inside ``reset()`` per episode so deck
draws are a pure function of ``(seed, episode_count)``.

These tests are engine-gated (they instantiate a real ``TrainingEnv``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.rl.conftest import make_deck_pool, requires_engine


@requires_engine
def test_training_env_max_steps_default_and_forwarding() -> None:
    """TrainingConfig defaults max_steps to 2000 and TrainingEnv forwards it to
    the underlying engine env."""
    from rl.config import TrainingConfig
    from rl.env_wrapper import TrainingEnv

    assert TrainingConfig().max_steps == 2000
    env = TrainingEnv(make_deck_pool(2), opponent="random", seed=42, max_steps=777)
    try:
        assert env._env._max_steps == 777
    finally:
        env.close()


@requires_engine
def test_step_no_auto_reset_on_done() -> None:
    """``step()`` on done returns terminal obs and leaves ``_episode_count`` unchanged."""
    from rl.env_wrapper import TrainingEnv

    deck_pool = make_deck_pool(2)
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="first",
    )
    try:
        obs = env.reset()
        ec_at_reset = env._episode_count
        # Drive the env to done — first legal action each step.
        for _ in range(800):
            action = int(np.argmax(obs["action_mask"]))
            obs, reward, done, info = env.step(action)
            if done:
                break
        else:
            pytest.skip("no done within 800 steps")

        assert done, "expected done flag"
        assert env._episode_count == ec_at_reset, (
            "step() incremented _episode_count on done — auto-reset still firing"
        )
        # Terminal-info fields are populated.
        assert "terminal_reward" in info
        assert "agent_deck_idx" in info
    finally:
        env.close()


@requires_engine
def test_reset_explicit_advances_counter() -> None:
    """Sequential ``reset()`` increments _episode_count by 1 each call."""
    from rl.env_wrapper import TrainingEnv

    deck_pool = make_deck_pool(2)
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="first",
    )
    try:
        env.reset()
        assert env._episode_count == 1
        env.reset()
        assert env._episode_count == 2
        env.reset()
        assert env._episode_count == 3
    finally:
        env.close()


@requires_engine
def test_reset_with_episode_idx_addresses_specific_episode() -> None:
    """``reset(episode_idx=3)`` produces the same deck draw as 3 sequential resets."""
    from rl.env_wrapper import TrainingEnv

    deck_pool = make_deck_pool(2)

    env_seq = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="random",
    )
    env_addr = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="random",
    )
    try:
        # env_seq: advance to episode 3 by sequential resets (no playing —
        # we only care about deck-RNG state and counter).
        env_seq.reset()
        env_seq.reset()
        env_seq.reset()
        seq_deck = env_seq._last_agent_deck_idx
        seq_ep_count = env_seq._episode_count

        # env_addr: jump directly to episode 3.
        env_addr.reset(episode_idx=3)
        addr_deck = env_addr._last_agent_deck_idx
        addr_ep_count = env_addr._episode_count

        assert seq_ep_count == addr_ep_count == 3
        assert seq_deck == addr_deck, (
            f"deck divergence: sequential={seq_deck}, addressed={addr_deck}"
        )
    finally:
        env_seq.close()
        env_addr.close()


@requires_engine
def test_reset_with_episode_idx_resequences() -> None:
    """``reset(episode_idx=N)`` is order-independent — same deck draw regardless of call history."""
    from rl.env_wrapper import TrainingEnv

    deck_pool = make_deck_pool(2)
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=42,
        agent_player="random",
    )
    try:
        env.reset(episode_idx=5)
        deck_5a = env._last_agent_deck_idx

        env.reset(episode_idx=2)

        env.reset(episode_idx=5)
        deck_5b = env._last_agent_deck_idx

        assert deck_5a == deck_5b, (
            f"reset(episode_idx=5) is non-deterministic: first={deck_5a}, second={deck_5b}"
        )
    finally:
        env.close()


@requires_engine
def test_deck_rng_reseeded_per_episode() -> None:
    """The deck pair is a pure function of (seed, episode), not of call order.

    Asserted against DeckSelector rather than against a reimplementation of the
    derivation, so the two cannot agree on a formula that is wrong.
    """
    from rl.deck_selector import DeckSelector
    from rl.env_wrapper import TrainingEnv

    deck_pool = make_deck_pool(2)
    seed = 42
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=seed,
        agent_player="first",
    )
    try:
        expected = DeckSelector(pool_size=len(deck_pool), seed=seed, allocation="random")
        # Out of order, and revisiting an episode, to pin that neither matters.
        for episode in (7, 3, 7):
            env.reset(episode_idx=episode)
            assert env._last_agent_deck_idx == expected.select(episode)[0], (
                f"episode {episode}: deck {env._last_agent_deck_idx} does not match "
                f"the selector's {expected.select(episode)[0]}"
            )
    finally:
        env.close()


def test_compute_advantage_reads_hand_counts_not_deck_counts() -> None:
    """Card-advantage shaping must compare HANDS.

    ``GlobalState`` comes from the real encoder rather than being
    hand-assembled, so the test names no field the encoder does not.
    Deck and hand counts are given distinct values in both directions, so
    reading a deck field cannot coincidentally produce the right answer.
    """
    from env.game_state import GameState
    from env.observation import build_observation
    from rl.env_wrapper import TrainingEnv

    gs = GameState()
    gs.deck_count = [30, 20]
    gs.hand_count = [5, 3]

    global_state = build_observation(gs, current_msg=None, agent_player=0)["global_state"]

    advantage = TrainingEnv._compute_advantage(global_state)
    assert advantage == 5 - 3, (
        f"expected the hand difference 2, got {advantage}; "
        f"{30 - 20} would mean it read the deck counts"
    )
