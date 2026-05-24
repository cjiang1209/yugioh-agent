"""Tests for opponent policies and seed determinism."""

import random
import tempfile

import numpy as np
import pytest

from yugioh_core.constants import MSG_SELECT_YESNO
from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_env.action_space import ActionMapper
from yugioh_env.opponent import GreedyOpponent, RandomOpponent


def _make_yesno_mapper() -> ActionMapper:
    """Create an ActionMapper with a simple yes/no message (2 actions)."""
    mapper = ActionMapper()
    mapper.update({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})
    return mapper


def test_random_opponent_deterministic_with_seed():
    """Same seed should produce identical action sequences."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    results = []
    for _ in range(2):
        opp = RandomOpponent(seed=42)
        mapper = _make_yesno_mapper()
        actions = [opp.select_action(msg, mapper.num_actions) for _ in range(20)]
        results.append(actions)
    assert results[0] == results[1]


def test_random_opponent_reseed_restores_determinism():
    """Calling reseed() should reset the RNG to produce the same sequence."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    opp = RandomOpponent(seed=99)
    mapper = _make_yesno_mapper()

    # Generate a sequence
    run1 = [opp.select_action(msg, mapper.num_actions) for _ in range(20)]

    # Reseed and generate again
    opp.reseed(99)
    run2 = [opp.select_action(msg, mapper.num_actions) for _ in range(20)]

    assert run1 == run2


def test_random_opponent_different_seeds_differ():
    """Different seeds should (almost certainly) produce different sequences."""
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0}
    mapper = _make_yesno_mapper()

    opp1 = RandomOpponent(seed=1)
    opp2 = RandomOpponent(seed=2)
    run1 = [opp1.select_action(msg, mapper.num_actions) for _ in range(50)]
    run2 = [opp2.select_action(msg, mapper.num_actions) for _ in range(50)]

    assert run1 != run2


def test_greedy_opponent_reseed_is_noop():
    """GreedyOpponent.reseed() should not raise."""
    opp = GreedyOpponent()
    opp.reseed(42)  # should be a no-op


def test_pick_action_random_seeded():
    """Client-side pick_action_random is deterministic when random module is seeded."""
    # Import here to avoid polluting module-level random state
    from cli.play_client import pick_action_random

    from yugioh_env.models import YuGiOhObservation

    mask = [1, 1, 1, 1, 0, 0, 0, 0] + [0] * 24  # 4 legal actions
    obs = YuGiOhObservation(
        cards=[],
        global_state=[0] * 20,
        actions=[[0] * 12] * 32,
        action_mask=mask,
        done=False,
        reward=0.0,
    )

    results = []
    for _ in range(2):
        random.seed(123)
        actions = [pick_action_random(obs) for _ in range(20)]
        results.append(actions)

    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Base Opponent defaults
# ---------------------------------------------------------------------------


def test_base_opponent_needs_observation_default():
    """Base Opponent.needs_observation returns False by default."""
    opp = RandomOpponent(seed=0)
    assert opp.needs_observation is False


def test_base_opponent_set_observation_is_noop():
    """Base Opponent.set_observation does nothing and doesn't raise."""
    opp = GreedyOpponent()
    opp.set_observation({"cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)})


# ---------------------------------------------------------------------------
# ModelOpponent tests (require torch + yugioh_rl)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _make_synthetic_checkpoint(path: str) -> None:
    """Create a minimal valid checkpoint file with default config."""
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import YuGiOhNet

    config = TrainingConfig()
    net = YuGiOhNet.from_config(config)
    torch.save(
        {
            "update": 1,
            "global_step": 100,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
        },
        path,
    )


def _dummy_obs() -> dict[str, np.ndarray]:
    """Create dummy observation arrays with valid shapes."""
    obs = {
        "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
        "global_state": np.zeros(GLOBAL_FEATURES, dtype=np.uint8),
        "actions": np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8),
        "action_mask": np.zeros(32, dtype=np.int8),
    }
    # Mark first 3 actions as legal
    obs["action_mask"][:3] = 1
    return obs


def test_model_opponent_construction():
    """ModelOpponent loads a checkpoint and enters eval mode."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")
        assert opp.needs_observation is True
        assert not opp._impl._network.training


def test_model_opponent_select_action():
    """ModelOpponent returns a valid action index within bounds."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")

        obs = _dummy_obs()
        opp.set_observation(obs)

        msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
        mapper = _make_yesno_mapper()
        action = opp.select_action(msg, mapper.num_actions)
        assert 0 <= action < mapper.num_actions


def test_model_opponent_deterministic():
    """Same checkpoint and observation should produce the same action."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")

        obs = _dummy_obs()
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
        mapper = _make_yesno_mapper()

        opp.set_observation(obs)
        a1 = opp.select_action(msg, mapper.num_actions)
        opp.set_observation(obs)
        a2 = opp.select_action(msg, mapper.num_actions)
        assert a1 == a2


def test_model_opponent_reseed_noop():
    """ModelOpponent.reseed() should not raise."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")
        opp.reseed(42)  # should be a no-op


def test_model_opponent_no_obs_returns_zero():
    """If set_observation was never called, select_action returns 0."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")

        msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
        mapper = _make_yesno_mapper()
        action = opp.select_action(msg, mapper.num_actions)
        assert action == 0


def test_model_opponent_semantic_checkpoint():
    """ModelOpponent works with a semantic-mode checkpoint (no embeddings file on disk)."""
    from yugioh_env.opponent import ModelOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import TextEmbeddingLookup, YuGiOhNet

    # Build a semantic-mode network from a synthetic embeddings file
    codes = list(range(1, 21))
    embeddings = torch.randn(len(codes), 384)
    codes_tensor = torch.tensor(codes, dtype=torch.int64)
    sorted_indices = codes_tensor.argsort()
    sorted_codes = codes_tensor[sorted_indices]
    sorted_embeddings = embeddings[sorted_indices]
    padded = torch.cat([torch.zeros(1, 384), sorted_embeddings], dim=0)

    text_lookup = TextEmbeddingLookup(sorted_codes, padded, text_embed_dim=32)
    config = TrainingConfig(text_embed_dim=32, learned_embed_dim=8)
    net = YuGiOhNet(config, text_lookup)

    # Save checkpoint (no embeddings file path in config)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {
                "update": 1,
                "global_step": 100,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": {},
                "config": config,
            },
            f.name,
        )
        ckpt_path = f.name

    # Load ModelOpponent — should NOT attempt to read an embeddings file
    opp = ModelOpponent(ckpt_path, device="cpu")
    assert opp.needs_observation is True
    assert opp._impl._network.text_lookup is not None

    # Verify select_action works
    obs = _dummy_obs()
    opp.set_observation(obs)
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
    mapper = _make_yesno_mapper()
    action = opp.select_action(msg, mapper.num_actions)
    assert 0 <= action < mapper.num_actions

    import os

    os.unlink(ckpt_path)


def test_model_opponent_env_config_missing_checkpoint():
    """opponent_type='model' without checkpoint should raise ValueError."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    with pytest.raises(ValueError, match="checkpoint path"):
        YuGiOhEnvironment(config={"opponent": "model:"})
