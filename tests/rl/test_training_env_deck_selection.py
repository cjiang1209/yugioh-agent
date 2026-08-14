from unittest.mock import patch

from tests.rl.conftest import make_fake_obs

POOL = [{"main": list(range(1, 41)), "extra": []} for _ in range(31)]


def test_mirror_gives_both_sides_same_deck():
    with patch("yugioh_env.server.yugioh_environment.YuGiOhEnvironment") as MockEnv:
        mock = MockEnv.return_value
        mock.reset.return_value = make_fake_obs()
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
        mock.reset.return_value = make_fake_obs()
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
