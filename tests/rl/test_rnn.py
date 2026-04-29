"""Tests for the recurrent-policy feature.

Phase 1 lands tests #7 and #8 from the plan: legacy-checkpoint resume and
legacy-checkpoint inference. Tests #1–#6, #10, #11 land in later phases.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet


_RNN_FIELDS = ("rnn_type", "rnn_hidden_dim", "rnn_num_layers", "bptt_chunk_len")


def _save_minimal_checkpoint(path: str, config: TrainingConfig) -> None:
    """Save just the keys ModelOpponent / ModelAgent / _build_resume_config
    actually read.  Drops update / global_step / optimizer_state_dict /
    episode_* lists carried by the resume-test helper — those are needed
    for full PPOTrainer resume but not for the lighter-weight rnn tests."""
    net = YuGiOhNet.from_config(config)
    torch.save({"config": config, "model_state_dict": net.state_dict()}, path)


def _make_legacy_checkpoint(path: str) -> None:
    """Save a checkpoint whose pickled config predates the RNN fields, by
    deleting the four RNN attributes from cfg.__dict__ before pickling."""
    config = TrainingConfig()
    for name in _RNN_FIELDS:
        del config.__dict__[name]
    _save_minimal_checkpoint(path, config)


def test_legacy_checkpoint_resume_backfills_rnn_fields(tmp_path, monkeypatch):
    """Plan test #7. Legacy ckpt missing the four RNN fields should resume
    cleanly, with each back-filled to its dataclass default."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "legacy.pt")
    _make_legacy_checkpoint(ckpt_path)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    cfg = _build_resume_config(args, str(tmp_path))

    assert cfg.rnn_type == "none"
    assert cfg.rnn_hidden_dim == 256
    assert cfg.rnn_num_layers == 1
    assert cfg.bptt_chunk_len == 16


def test_legacy_checkpoint_inference_via_model_opponent(tmp_path):
    """Plan test #8 (ModelOpponent half).  Legacy ckpt should load and run
    inference without AttributeError on the new RNN fields."""
    from yugioh_core.constants import MSG_SELECT_YESNO
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "legacy.pt")
    _make_legacy_checkpoint(ckpt_path)

    opp = ModelOpponent(ckpt_path, device="cpu")

    obs = {
        "cards": np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.zeros(20, dtype=np.uint8),
        "actions": np.zeros((32, 12), dtype=np.uint8),
        "action_mask": np.zeros(32, dtype=np.int8),
    }
    obs["action_mask"][:3] = 1
    opp.set_observation(obs)

    msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
    action = opp.select_action(msg, num_actions=3)
    assert 0 <= action < 3


def test_legacy_checkpoint_inference_via_model_agent(tmp_path, db_path):
    """Plan test #8 (ModelAgent half).  Legacy ckpt should load via the
    MUD-bot ModelAgent without AttributeError."""
    from yugioh_mud.agent import ModelAgent

    ckpt_path = str(tmp_path / "legacy.pt")
    _make_legacy_checkpoint(ckpt_path)

    agent = ModelAgent(ckpt_path, str(db_path), device="cpu")
    assert not agent._network.training


# ---------------------------------------------------------------------------
# Phase 2: feed-forward parity + RNN checkpoint round-trip
# ---------------------------------------------------------------------------


def _dummy_obs_tensors(batch: int = 4):
    cards = torch.zeros(batch, 200, 42, dtype=torch.uint8)
    glob = torch.zeros(batch, 20, dtype=torch.uint8)
    actions = torch.zeros(batch, 32, 12, dtype=torch.uint8)
    mask = torch.ones(batch, 32, dtype=torch.int8)
    return cards, glob, actions, mask


def test_feed_forward_state_dict_unchanged_at_rnn_none():
    """Plan test #1 (key-set half).  rnn_type='none' must produce a state
    dict with no rnn.* keys, so pre-RNN checkpoints stay byte-identical."""
    config = TrainingConfig()  # rnn_type defaults to "none"
    net = YuGiOhNet.from_config(config)
    keys = set(net.state_dict().keys())
    assert net.rnn is None
    assert not any(k.startswith("rnn.") for k in keys), \
        f"rnn_type='none' should not emit rnn.* keys; got: {sorted(keys)}"


def test_feed_forward_outputs_deterministic_at_rnn_none():
    """Plan test #1 (output half).  Two networks built with the same seed
    and rnn_type='none' produce identical logits and values for the same
    input — i.e. the rnn=None branch is the original feed-forward path."""
    config = TrainingConfig()
    cards, glob, actions, mask = _dummy_obs_tensors(batch=2)

    torch.manual_seed(0)
    net_a = YuGiOhNet.from_config(config)
    torch.manual_seed(0)
    net_b = YuGiOhNet.from_config(config)

    net_a.eval()
    net_b.eval()
    with torch.no_grad():
        la, va, hxa = net_a(cards, glob, actions, mask)
        lb, vb, hxb = net_b(cards, glob, actions, mask)

    assert hxa is None and hxb is None
    # action_mask=1 everywhere → no -inf masking, so vanilla allclose works.
    assert torch.allclose(la, lb)
    assert torch.allclose(va, vb)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_rnn_checkpoint_roundtrip_preserves_outputs(tmp_path, rnn_type):
    """Plan test #5 (shape-level + bit-equality on inference).  A network
    built with an RNN should round-trip through state_dict + from_state_dict
    and produce identical (logits, values, new_hx) for the same input + hx."""
    config = TrainingConfig(rnn_type=rnn_type, rnn_hidden_dim=64, rnn_num_layers=1)
    torch.manual_seed(0)
    net = YuGiOhNet.from_config(config)
    net.eval()

    cards, glob, actions, mask = _dummy_obs_tensors(batch=3)
    hx = net.init_hx(batch_size=3, device=torch.device("cpu"))

    with torch.no_grad():
        logits_ref, values_ref, hx_ref = net(cards, glob, actions, mask, hx=hx)

    # Round-trip through the on-disk state dict.
    ckpt_path = str(tmp_path / f"rnn_{rnn_type}.pt")
    torch.save({"config": config, "model_state_dict": net.state_dict()}, ckpt_path)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded = YuGiOhNet.from_state_dict(blob["config"], blob["model_state_dict"])
    reloaded.eval()

    with torch.no_grad():
        logits_new, values_new, hx_new = reloaded(cards, glob, actions, mask, hx=hx)

    assert torch.allclose(logits_ref, logits_new)
    assert torch.allclose(values_ref, values_new)
    if rnn_type == "lstm":
        assert torch.allclose(hx_ref[0], hx_new[0])
        assert torch.allclose(hx_ref[1], hx_new[1])
    else:
        assert torch.allclose(hx_ref, hx_new)


# ---------------------------------------------------------------------------
# Phase 3: hidden-state threading through collection + ModelOpponent lifecycle
# ---------------------------------------------------------------------------


def test_mask_hx_zeros_only_done_envs():
    """Plan test #3 (focused on mask_hx).  Hand-crafted dones over 4 steps
    × 2 envs; simulate the collection-loop pattern with an identity-increment
    cell and assert each env's hx is zero at exactly the steps following done.
    """
    num_envs = 2
    hidden = 3
    num_layers = 1

    hx = torch.zeros(num_layers, num_envs, hidden)
    dones_per_step = torch.tensor(
        [[0, 0],
         [1, 0],
         [0, 0],
         [0, 1]],
        dtype=torch.float32,
    )

    history = []
    for t in range(dones_per_step.shape[0]):
        # "Identity-increment" cell: each step adds 1 to every hx entry.
        hx = hx + 1.0
        history.append(hx.clone())
        hx = YuGiOhNet.mask_hx(hx, dones_per_step[t])

    # After step 0: both envs accumulated +1.
    assert torch.equal(history[0][0, 0], torch.full((hidden,), 1.0))
    assert torch.equal(history[0][0, 1], torch.full((hidden,), 1.0))
    # Step 1: env 0 incremented to +2 then masked to 0; env 1 stays at +2 (next step starts from +2).
    assert torch.equal(history[1][0, 0], torch.full((hidden,), 2.0))
    assert torch.equal(history[1][0, 1], torch.full((hidden,), 2.0))
    # Step 2: env 0 starts from 0 → +1; env 1 starts from +2 → +3.
    assert torch.equal(history[2][0, 0], torch.full((hidden,), 1.0))
    assert torch.equal(history[2][0, 1], torch.full((hidden,), 3.0))
    # Step 3: env 0 → +2; env 1 → +4.
    assert torch.equal(history[3][0, 0], torch.full((hidden,), 2.0))
    assert torch.equal(history[3][0, 1], torch.full((hidden,), 4.0))
    # Final hx (after step-3 mask): env 0 keeps +2, env 1 done → 0.
    assert torch.equal(hx[0, 0], torch.full((hidden,), 2.0))
    assert torch.equal(hx[0, 1], torch.zeros(hidden))


def test_mask_hx_handles_lstm_tuple():
    """Both halves of an LSTM (h, c) tuple should be masked together."""
    h = torch.ones(1, 2, 3)
    c = torch.full((1, 2, 3), 5.0)
    dones = torch.tensor([1.0, 0.0])
    h_new, c_new = YuGiOhNet.mask_hx((h, c), dones)
    assert torch.equal(h_new[0, 0], torch.zeros(3))
    assert torch.equal(h_new[0, 1], torch.ones(3))
    assert torch.equal(c_new[0, 0], torch.zeros(3))
    assert torch.equal(c_new[0, 1], torch.full((3,), 5.0))


def _make_rnn_checkpoint(path: str, rnn_type: str = "lstm") -> TrainingConfig:
    """Save an RNN-mode ckpt and return the config used to build it."""
    config = TrainingConfig(rnn_type=rnn_type, rnn_hidden_dim=64, rnn_num_layers=1)
    _save_minimal_checkpoint(path, config)
    return config


def test_model_opponent_hx_lifecycle(tmp_path):
    """Plan test #6.  Instantiate ModelOpponent on an RNN ckpt, run a few
    select_action calls; assert _hx is non-None and changes between calls.
    """
    from yugioh_core.constants import MSG_SELECT_YESNO
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "rnn.pt")
    _make_rnn_checkpoint(ckpt_path, rnn_type="lstm")

    opp = ModelOpponent(ckpt_path, device="cpu")

    obs = {
        "cards": np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.zeros(20, dtype=np.uint8),
        "actions": np.zeros((32, 12), dtype=np.uint8),
        "action_mask": np.zeros(32, dtype=np.int8),
    }
    obs["action_mask"][:3] = 1

    # Reseed initialises hx to zero.
    opp.reseed(0)
    inner = opp._impl
    assert inner._hx is not None
    h0, c0 = inner._hx
    assert torch.equal(h0, torch.zeros_like(h0))
    assert torch.equal(c0, torch.zeros_like(c0))

    # Call select_action; hx should advance away from zero.
    msg = {"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}
    opp.set_observation(obs)
    opp.select_action(msg, num_actions=3)
    h1, c1 = inner._hx
    assert not torch.equal(h1, h0) or not torch.equal(c1, c0), \
        "select_action must advance hx for an RNN-mode network"

    # A second call should advance hx again.
    opp.set_observation(obs)
    opp.select_action(msg, num_actions=3)
    h2, c2 = inner._hx
    assert not torch.equal(h2, h1) or not torch.equal(c2, c1), \
        "consecutive select_action calls must not produce identical hx"

    # Reseed clears hx back to zero.
    opp.reseed(7)
    h3, c3 = inner._hx
    assert torch.equal(h3, torch.zeros_like(h3))
    assert torch.equal(c3, torch.zeros_like(c3))


def test_model_opponent_feed_forward_hx_is_none(tmp_path):
    """Default rnn_type='none' ckpt: _hx must remain None across calls."""
    from yugioh_core.constants import MSG_SELECT_YESNO
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "ff.pt")
    _save_minimal_checkpoint(ckpt_path, TrainingConfig())

    opp = ModelOpponent(ckpt_path, device="cpu")
    opp.reseed(0)
    assert opp._impl._hx is None

    obs = {
        "cards": np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.zeros(20, dtype=np.uint8),
        "actions": np.zeros((32, 12), dtype=np.uint8),
        "action_mask": np.zeros(32, dtype=np.int8),
    }
    obs["action_mask"][:3] = 1
    opp.set_observation(obs)
    opp.select_action({"msg_type": MSG_SELECT_YESNO, "player": 1, "desc": 0}, num_actions=3)
    assert opp._impl._hx is None


def test_rollout_loop_resets_hx_per_rollout():
    """The trainer must call init_hx at the start of every rollout, even when
    the previous rollout ended with a non-zero hx.  Carrying hx across an
    optimizer step would feed pre-update-weights hx into the post-update
    network on step 0, breaking policy/value consistency.
    """
    import re
    import inspect
    from yugioh_rl.ppo import PPOTrainer

    src = inspect.getsource(PPOTrainer.train)
    # Locate the per-rollout for loop and the init_hx assignment.
    for_match = re.search(r"for update in range\(", src)
    init_match = re.search(r"hx = self\.network\.init_hx\(", src)
    assert for_match is not None, "PPOTrainer.train must contain the rollout for-loop"
    assert init_match is not None, "PPOTrainer.train must call self.network.init_hx"
    assert init_match.start() > for_match.start(), (
        "init_hx must live INSIDE the per-update for-loop so each rollout "
        "starts with a fresh hidden state. Hoisting it outside reintroduces "
        "the cross-rollout staleness bug."
    )


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_ppo_trainer_refuses_recurrent_training(tmp_path, rnn_type):
    """Until the TBPTT update path lands, PPOTrainer must hard-error on
    rnn_type != 'none'.  The rollout collects recurrently but the update
    replays feed-forward — silently off-policy if we let it run.
    """
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        rnn_type=rnn_type,
        rnn_hidden_dim=64,
        save_dir=str(tmp_path),
        num_envs=1,
    )
    with pytest.raises(NotImplementedError, match="rnn_type"):
        PPOTrainer(config)


def test_rnn_state_dict_mismatch_rejected_by_from_state_dict():
    """Defensive guard in from_state_dict: a (config, state_dict) pair where
    rnn_type='none' but the dict carries rnn.* keys (or vice versa) should
    raise — silent acceptance would corrupt training quality."""
    rnn_config = TrainingConfig(rnn_type="lstm", rnn_hidden_dim=64)
    rnn_net = YuGiOhNet.from_config(rnn_config)
    rnn_state = rnn_net.state_dict()

    none_config = TrainingConfig(rnn_type="none")
    with pytest.raises(ValueError, match="rnn"):
        YuGiOhNet.from_state_dict(none_config, rnn_state)

    none_net = YuGiOhNet.from_config(none_config)
    with pytest.raises(ValueError, match="rnn"):
        YuGiOhNet.from_state_dict(rnn_config, none_net.state_dict())
