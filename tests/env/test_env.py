"""Full environment integration tests."""

import pytest

from yugioh_env.deck_parser import parse_ydk
from yugioh_env.models import YuGiOhAction
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment


@pytest.fixture
def env(lib, db_path, script_dirs, deck_path):
    """Create a YuGiOhEnvironment instance."""
    config = {
        "lib_path": None,  # auto-detect
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
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
    """Parse the default deck file into an inline dict."""
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


# --- Agent player order tests ---


def test_agent_player_invalid_config(lib, db_path, script_dirs, deck_path):
    """Invalid agent_player config value should raise ValueError."""
    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "agent_player": 2,
    }
    with pytest.raises(ValueError, match="agent_player"):
        YuGiOhEnvironment(config)


def test_reset_agent_player_go_second(env):
    """Agent goes second when agent_player=1 is passed to reset."""
    obs = env.reset(seed=42, agent_player=1)
    assert obs is not None
    assert not obs.done
    assert env._agent_player == 1
    assert any(a == 1 for a in obs.action_mask)


def test_reset_agent_player_explicit_zero(env):
    """Explicit agent_player=0 behaves like default (go first)."""
    obs = env.reset(seed=42, agent_player=0)
    assert obs is not None
    assert env._agent_player == 0


def test_reset_agent_player_random_deterministic(env):
    """Random agent_player with same seed produces same choice."""
    env.reset(seed=42, agent_player="random")
    choice1 = env._agent_player
    env.reset(seed=42, agent_player="random")
    choice2 = env._agent_player
    assert choice1 == choice2


def test_reset_agent_player_config_default(lib, db_path, script_dirs, deck_path):
    """Config-level agent_player is used when reset doesn't override."""
    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "agent_player": 1,
    }
    e = YuGiOhEnvironment(config)
    try:
        obs = e.reset(seed=42)
        assert e._agent_player == 1
        assert not obs.done
    finally:
        e.close()


def test_reset_agent_player_override(lib, db_path, script_dirs, deck_path):
    """Per-reset agent_player overrides config default."""
    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "agent_player": 0,
    }
    e = YuGiOhEnvironment(config)
    try:
        e.reset(seed=42, agent_player=1)
        assert e._agent_player == 1
        # Next reset without override goes back to config default
        e.reset(seed=43)
        assert e._agent_player == 0
    finally:
        e.close()


def test_full_episode_go_second(env):
    """Play a full episode with agent as player 1 (going second)."""
    obs = env.reset(seed=42, agent_player=1)
    steps = 0
    max_steps = 500

    while not obs.done and steps < max_steps:
        action_idx = 0
        for i, mask in enumerate(obs.action_mask):
            if mask == 1:
                action_idx = i
                break
        obs = env.step(YuGiOhAction(action_index=action_idx))
        steps += 1

    if obs.done:
        assert obs.reward in (1.0, -1.0, 0.0)


def test_obs_events_normal_step_matches_cycle(env):
    obs = env.reset(seed=42)
    assert obs.events == list(env._cycle_events)


def test_obs_events_empty_on_multi_select_substep(env):
    from yugioh_core.constants import MSG_SELECT_IDLECMD, MSG_SELECT_TRIBUTE

    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": 89631139, "pos": "face_up_attack", "seq": 0},
                {"code": 89631139, "pos": "face_up_attack", "seq": 1},
            ],
            "hand": [46986414],
            "deck": [89631139],
        }
    }
    env.reset(puzzle=puzzle)
    assert env._mapper.msg_type == MSG_SELECT_IDLECMD
    env.step(YuGiOhAction(action_index=0))
    assert env._mapper.msg_type == MSG_SELECT_TRIBUTE
    obs = env.step(YuGiOhAction(action_index=0))
    assert obs.events == []


def test_make_observation_raises_on_zero_actions(env):
    """An agent-facing prompt with no legal actions (e.g. an unenumerable
    MSG_ANNOUNCE_CARD general-predicate filter) must fail with a diagnosable
    error naming the msg_type, not an opaque NaN downstream."""
    from yugioh_core.constants import MSG_ANNOUNCE_CARD

    env.reset(seed=42)

    # General-predicate filter: the first opcode is an opcode word (>= 0x4000...),
    # not a literal card code, so _parse_announce_codes yields [] -> 0 actions.
    msg = {
        "msg_type": MSG_ANNOUNCE_CARD,
        "player": env._agent_player,
        "opcodes": [0x4000020000000000, 0x10],
    }
    env._current_msg = msg
    env._mapper.update({**msg, "_agent_player": env._agent_player})
    assert env._mapper.num_actions == 0

    with pytest.raises(RuntimeError, match=str(MSG_ANNOUNCE_CARD)):
        env._make_observation()


def test_duelview_no_duel():
    """The properties still answer without a duel; query_location refuses."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment.__new__(YuGiOhEnvironment)  # no engine boot
    env._duel = None
    env._agent_player = 0
    sentinel = object()
    env._card_db = sentinel
    assert env.game_state is None
    with pytest.raises(AssertionError):
        env.query_location(0, 1)
    assert env.is_finished is True
    assert env.winner is None
    assert env.agent_player == 0
    assert env.card_db is sentinel


def test_duelview_delegates_live(env):
    env.reset(seed=42)  # fixture only CONSTRUCTS — reset first
    assert env.game_state is not None
    assert env.agent_player in (0, 1)
    assert isinstance(env.query_location(env.agent_player, 1), list)
    assert env.is_finished in (True, False)
