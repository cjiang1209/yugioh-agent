"""Test observation encoding.

Encode→decode roundtrip tests live in test_feature_roundtrip.py.
This file covers observation-building logic that doesn't go through
the decoder (e.g. query_fn integration, visibility rules).
"""


def test_action_meta_length_matches_actions(lib, db_path, script_dirs):
    """action_meta length must equal action_mask length (32 for active obs).
    This is the §6 length-parity invariant from the spec."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    assert len(obs.action_meta) == len(obs.actions) == len(obs.action_mask)
    # Inactive slots are None; only the legal-action prefix may carry meta
    legal_count = sum(obs.action_mask)
    for i in range(legal_count, len(obs.action_mask)):
        assert obs.action_meta[i] is None


def test_terminal_observation_lists_empty(lib, db_path, script_dirs):
    """On done=True, actions, action_mask, and action_meta are all empty.
    This is an intentional drift from the previous action_mask=[0]*32 behavior
    (§3 of spec) — kept consistent so the three lists never differ in length."""
    import random
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    from yugioh_env.models import YuGiOhAction
    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    rng = random.Random(0)
    while not obs.done:
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))
    assert obs.done
    assert obs.actions == []
    assert obs.action_mask == []
    assert obs.action_meta == []
