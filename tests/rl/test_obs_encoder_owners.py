"""The two network owners must feed the policy through the shared encoder, so a
layout change cannot reach one and miss the other.

These assert the call happens, not that a name appears in the source: a
substring check passes on a comment mentioning the function.

Patch targets differ by import style. TrainingEnv imports encode_observation at
module top, so the binding to replace lives on env_wrapper. NetworkOpponent
imports it inside select_action -- torch is an optional [train] extra -- so
that name resolves at call time and the source module is what must be
patched.
"""

from __future__ import annotations

import pytest

from tests.rl.conftest import make_deck_pool, requires_engine


def _spy_encode(monkeypatch, module) -> list[object]:
    """Record every observation the module's encode_observation binding sees.

    The module is a parameter because the two owners bind the name at
    different times, which is what these tests are about.
    """
    calls: list[object] = []
    real = module.encode_observation

    def spy(obs):
        calls.append(obs)
        return real(obs)

    monkeypatch.setattr(module, "encode_observation", spy)
    return calls


@requires_engine
def test_training_env_encodes_through_the_shared_encoder(monkeypatch) -> None:
    import yugioh_rl.env_wrapper as ew
    from yugioh_rl.env_wrapper import TrainingEnv

    calls = _spy_encode(monkeypatch, ew)

    env = TrainingEnv(make_deck_pool(), opponent="random", seed=0)
    try:
        env.reset()
        assert calls, "TrainingEnv.reset did not encode through encode_observation"
        before = len(calls)
        env.step(0)
        assert len(calls) > before, "TrainingEnv.step did not encode through encode_observation"
    finally:
        env.close()


def test_network_opponent_encodes_through_the_shared_encoder(monkeypatch) -> None:
    pytest.importorskip("torch")

    import yugioh_rl.obs_encoder as oe
    from tests.env.conftest import MINIMAL_MSGS, obs_from_msg
    from yugioh_core.constants import MSG_SELECT_YESNO
    from yugioh_env.opponent import NetworkOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import YuGiOhNet

    calls = _spy_encode(monkeypatch, oe)

    net = YuGiOhNet.from_config(TrainingConfig())
    net.eval()
    opponent = NetworkOpponent(net, device="cpu")
    obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_YESNO], "msg_type": MSG_SELECT_YESNO})
    opponent.select_action(obs)
    assert calls, "NetworkOpponent.select_action did not encode through encode_observation"
