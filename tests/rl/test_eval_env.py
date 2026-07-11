from unittest.mock import MagicMock, patch

from yugioh_core.encoding import GLOBAL_FEATURES


def _obs(done=False, reward=0.0):
    o = MagicMock()
    o.cards = [0] * (200 * 42)
    o.global_state = [0] * GLOBAL_FEATURES
    o.actions = [0] * (32 * 28)
    o.action_mask = [0] * 32
    o.pending_chain = []
    o.event_history = []
    o.reward = reward
    o.done = done
    return o


POOL = [{"main": list(range(1, 41)), "extra": []} for _ in range(31)]


def test_eval_env_deck_idx_in_terminal_info():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock = MockEnv.return_value
        mock.reset.return_value = _obs()
        mock.step.return_value = _obs(done=True, reward=1.0)
        mock._step_count = 7
        from yugioh_rl.env_wrapper import EvalEnv

        env = EvalEnv(
            deck_pool=POOL,
            opponent="random",
            seed=42,
            agent_player="first",
            deck_allocation="balanced",
        )
        env.reset(episode_idx=1)  # balanced → agent deck 0
        _, _, done, info = env.step(0)
        assert done is True
        assert info["terminal_reward"] == 1.0
        assert info["episode_length"] == 7
        assert info["agent_deck_idx"] == 0


def test_eval_env_has_no_shaping_or_pool():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        MockEnv.return_value.reset.return_value = _obs()
        from yugioh_rl.env_wrapper import EvalEnv

        env = EvalEnv(deck_pool=POOL, opponent="random", seed=42)
        assert not hasattr(env, "_opponent_pool")
        assert not hasattr(env, "_reward_shaping")


def test_eval_env_mirror_same_deck_object():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock = MockEnv.return_value
        mock.reset.return_value = _obs()
        from yugioh_rl.env_wrapper import EvalEnv

        env = EvalEnv(
            deck_pool=POOL, opponent="random", seed=42, agent_player="first", mirror_decks=True
        )
        env.reset(episode_idx=3)
        call = mock.reset.call_args
        assert call.kwargs["deck0"] is call.kwargs["deck1"]
