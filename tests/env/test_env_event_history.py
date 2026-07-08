import numpy as np


def _make_env(db_path, script_dirs, deck_path):
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
    }
    return YuGiOhEnvironment(config)


def test_event_history_shape_in_obs(lib, db_path, script_dirs, deck_path):
    from yugioh_core.encoding import EVENT_ENTRY_FEATURES, MAX_EVENT_HISTORY

    env = _make_env(db_path, script_dirs, deck_path)
    obs = env.reset(seed=42, agent_player=0)
    ev = np.array(obs.event_history, dtype=np.uint8)
    assert ev.shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    env.close()


def test_event_buffer_resets_between_episodes(lib, db_path, script_dirs, deck_path):
    from yugioh_core.encoding import MAX_EVENT_HISTORY
    from yugioh_env.models import YuGiOhAction

    env = _make_env(db_path, script_dirs, deck_path)
    obs = env.reset(seed=1, agent_player=0)
    steps = 0
    while not obs.done and steps < 6:
        obs = env.step(YuGiOhAction(action_index=0))
        steps += 1
    obs2 = env.reset(seed=2, agent_player=0)
    ev2 = np.array(obs2.event_history, dtype=np.uint8)
    # New episode observation must carry a correctly-shaped event_history and
    # never a full/overflowed buffer inherited from the prior episode.
    assert ev2.shape[0] == MAX_EVENT_HISTORY
    env.close()
