from unittest.mock import MagicMock, patch

from yugioh_core.encoding import GLOBAL_FEATURES


def _fake_obs():
    obs = MagicMock()
    obs.cards = [0] * (200 * 42)
    obs.global_state = [0] * GLOBAL_FEATURES
    obs.actions = [0] * (32 * 28)
    obs.action_mask = [0] * 32
    obs.pending_chain = []
    obs.event_history = []
    obs.reward = 0.0
    obs.done = False
    return obs


POOL = [{"main": list(range(1, 41)), "extra": []} for _ in range(31)]


def test_mirror_gives_both_sides_same_deck():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock = MockEnv.return_value
        mock.reset.return_value = _fake_obs()
        mock._agent_player = 0
        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(
            deck_pool=POOL, seed=42, agent_player="first", mirror_decks=True, reward_shaping=False
        )
        env.reset()
        call = mock.reset.call_args
        assert call.kwargs["deck0"] is call.kwargs["deck1"]  # same object


def test_balanced_agent_deck_is_round_robin():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock = MockEnv.return_value
        mock.reset.return_value = _fake_obs()
        mock._agent_player = 0
        from yugioh_rl.env_wrapper import TrainingEnv

        env = TrainingEnv(
            deck_pool=POOL,
            seed=42,
            agent_player="first",
            deck_allocation="balanced",
            reward_shaping=False,
        )
        idxs = []
        for ep in range(1, 5):
            env.reset(episode_idx=ep)
            idxs.append(env._last_agent_deck_idx)
        assert idxs == [0, 1, 2, 3]
