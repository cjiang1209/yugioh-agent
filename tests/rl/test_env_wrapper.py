"""Unit tests for the ``TrainingEnv`` lifecycle changes.

After the auto-reset removal, ``step()`` returns terminal obs on done and
``reset()`` accepts an optional ``episode_idx`` so episodes are addressable
by index.  ``_deck_rng`` is reseeded inside ``reset()`` per episode so deck
draws are a pure function of ``(seed, episode_count)``.

These tests are engine-gated (they instantiate a real ``TrainingEnv``).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from tests.rl.conftest import requires_engine
from yugioh_rl.env_wrapper import parse_deck_pool


def _make_deck_pool() -> list[dict[str, list[int]]]:
    deck_path = Path("assets/decks/blue_eyes.ydk")
    if not deck_path.exists():
        pytest.skip(f"missing deck: {deck_path}")
    # Two-deck pool so deck_rng draws can vary; both pointing at the same
    # file is fine — only the index matters for determinism tests.
    return parse_deck_pool([str(deck_path), str(deck_path)])


@requires_engine
def test_training_env_max_steps_default_and_forwarding() -> None:
    """TrainingConfig defaults max_steps to 2000 and TrainingEnv forwards it to
    the underlying engine env."""
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import TrainingEnv

    assert TrainingConfig().max_steps == 2000
    env = TrainingEnv(_make_deck_pool(), opponent="random", seed=42, max_steps=777)
    try:
        assert env._env._max_steps == 777
    finally:
        env.close()


@requires_engine
def test_step_no_auto_reset_on_done() -> None:
    """``step()`` on done returns terminal obs and leaves ``_episode_count`` unchanged."""
    from yugioh_rl.env_wrapper import TrainingEnv

    deck_pool = _make_deck_pool()
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
    from yugioh_rl.env_wrapper import TrainingEnv

    deck_pool = _make_deck_pool()
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
    from yugioh_rl.env_wrapper import TrainingEnv

    deck_pool = _make_deck_pool()

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
    from yugioh_rl.env_wrapper import TrainingEnv

    deck_pool = _make_deck_pool()
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
    """Deck pair at episode N matches ``random.Random(seed + N)`` — pure function of (seed, N)."""
    from yugioh_rl.env_wrapper import TrainingEnv

    deck_pool = _make_deck_pool()
    seed = 42
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        reward_shaping=False,
        seed=seed,
        agent_player="first",
    )
    try:
        env.reset(episode_idx=7)
        actual_deck = env._last_agent_deck_idx

        # Reproduce what reset() should have done internally.
        episode_seed = seed + 7
        rng = random.Random(episode_seed)
        expected_agent_deck = rng.randrange(len(deck_pool))
        # rng.randrange for opp_deck consumed too — but we only stored agent
        _ = rng.randrange(len(deck_pool))

        assert actual_deck == expected_agent_deck, (
            f"deck draw {actual_deck} != expected {expected_agent_deck} "
            f"from random.Random({episode_seed})"
        )
    finally:
        env.close()


def test_compute_advantage_reads_hand_counts_not_deck_counts() -> None:
    """Card-advantage shaping must compare HANDS.

    The global_state offsets are derived from the real encoder rather than
    hardcoded, so a future change to the layout moves this test with it
    instead of leaving it asserting stale slots. Deck and hand counts are
    given distinct values in both directions, so reading a deck slot cannot
    coincidentally produce the right answer.
    """
    from yugioh_env.game_state import GameState
    from yugioh_env.observation import build_observation
    from yugioh_rl.env_wrapper import TrainingEnv

    gs = GameState()
    gs.deck_count = [30, 20]
    gs.hand_count = [5, 3]

    packed = build_observation(gs, current_msg=None, agent_player=0)["global_state"]

    advantage = TrainingEnv._compute_advantage(packed)
    assert advantage == 5 - 3, (
        f"expected the hand difference 2, got {advantage}; "
        f"{30 - 20} would mean it read the deck counts"
    )
