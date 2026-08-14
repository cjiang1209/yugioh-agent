"""Tests for the recurrent-policy feature.

Phase 1 lands tests #7 and #8 from the plan: legacy-checkpoint resume and
legacy-checkpoint inference. Tests #1–#6, #10, #11 land in later phases.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_env.models import Pass, YuGiOhObservation
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet

_RNN_FIELDS = ("rnn_type", "rnn_hidden_dim", "rnn_num_layers", "bptt_chunk_len")


def _dummy_model_opponent_obs() -> YuGiOhObservation:
    """Full-shape observation with 3 legal actions."""
    return YuGiOhObservation(action_descriptors=[Pass() for _ in range(3)])


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
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "legacy.pt")
    _make_legacy_checkpoint(ckpt_path)

    opp = ModelOpponent(ckpt_path, device="cpu")

    action, _ = opp.select_action(_dummy_model_opponent_obs())
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
    cards = torch.zeros(batch, MAX_CARDS, CARD_FEATURES, dtype=torch.uint8)
    glob = torch.zeros(batch, GLOBAL_FEATURES, dtype=torch.uint8)
    actions = torch.zeros(batch, MAX_ACTIONS, ACTION_FEATURES, dtype=torch.uint8)
    mask = torch.ones(batch, MAX_ACTIONS, dtype=torch.int8)
    return cards, glob, actions, mask


def test_feed_forward_state_dict_unchanged_at_rnn_none():
    """Plan test #1 (key-set half).  rnn_type='none' must produce a state
    dict with no rnn.* keys, so pre-RNN checkpoints stay byte-identical."""
    config = TrainingConfig()  # rnn_type defaults to "none"
    net = YuGiOhNet.from_config(config)
    keys = set(net.state_dict().keys())
    assert net.rnn is None
    assert not any(k.startswith("rnn.") for k in keys), (
        f"rnn_type='none' should not emit rnn.* keys; got: {sorted(keys)}"
    )


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
        [[0, 0], [1, 0], [0, 0], [0, 1]],
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
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "rnn.pt")
    _make_rnn_checkpoint(ckpt_path, rnn_type="lstm")

    opp = ModelOpponent(ckpt_path, device="cpu")

    obs = _dummy_model_opponent_obs()

    # Reseed initialises hx to zero.
    opp.reseed(0)
    inner = opp._impl
    assert inner._hx is not None
    h0, c0 = inner._hx
    assert torch.equal(h0, torch.zeros_like(h0))
    assert torch.equal(c0, torch.zeros_like(c0))

    # Call select_action; hx should advance away from zero.
    opp.select_action(obs)
    h1, c1 = inner._hx
    assert not torch.equal(h1, h0) or not torch.equal(c1, c0), (
        "select_action must advance hx for an RNN-mode network"
    )

    # A second call should advance hx again.
    opp.select_action(obs)
    h2, c2 = inner._hx
    assert not torch.equal(h2, h1) or not torch.equal(c2, c1), (
        "consecutive select_action calls must not produce identical hx"
    )

    # Reseed clears hx back to zero.
    opp.reseed(7)
    h3, c3 = inner._hx
    assert torch.equal(h3, torch.zeros_like(h3))
    assert torch.equal(c3, torch.zeros_like(c3))


def test_model_opponent_feed_forward_hx_is_none(tmp_path):
    """Default rnn_type='none' ckpt: _hx must remain None across calls."""
    from yugioh_env.opponent import ModelOpponent

    ckpt_path = str(tmp_path / "ff.pt")
    _save_minimal_checkpoint(ckpt_path, TrainingConfig())

    opp = ModelOpponent(ckpt_path, device="cpu")
    opp.reseed(0)
    assert opp._impl._hx is None

    opp.select_action(_dummy_model_opponent_obs())
    assert opp._impl._hx is None


def test_rollout_loop_resets_hx_per_rollout():
    """The trainer must call init_hx at the start of every rollout, even when
    the previous rollout ended with a non-zero hx.  Carrying hx across an
    optimizer step would feed pre-update-weights hx into the post-update
    network on step 0, breaking policy/value consistency.
    """
    import inspect
    import re

    from yugioh_rl.ppo import PPOTrainer

    src = inspect.getsource(PPOTrainer.train)
    # Locate the per-rollout for loop and the init_hx assignment.
    for_match = re.search(r"for update in range\(", src)
    init_match = re.search(r"self\.network\.init_hx\(", src)
    assert for_match is not None, "PPOTrainer.train must contain the rollout for-loop"
    assert init_match is not None, "PPOTrainer.train must call self.network.init_hx"
    assert init_match.start() > for_match.start(), (
        "init_hx must live INSIDE the per-update for-loop so each rollout "
        "starts with a fresh hidden state. Hoisting it outside reintroduces "
        "the cross-rollout staleness bug."
    )


# ---------------------------------------------------------------------------
# Phase 4: TBPTT update path + checkpoint compat + config validation
# ---------------------------------------------------------------------------


def _make_tbptt_trainer(*, rollout_steps: int, num_envs: int, **overrides):
    """Build a PPOTrainer in RNN mode with a populated random rollout buffer.

    Used by tests that exercise ``_run_update_tbptt`` directly without
    spinning up a vec env.  Default overrides give ``minibatch_size ==
    rollout_steps`` (one minibatch per env) and ``num_epochs=1``; pass
    them in ``overrides`` to change either.
    """
    from yugioh_rl.ppo import PPOTrainer

    defaults = {
        "rnn_type": "lstm",
        "rnn_hidden_dim": 64,
        "rnn_num_layers": 1,
        "bptt_chunk_len": 8,
        "rollout_steps": rollout_steps,
        "num_envs": num_envs,
        "minibatch_size": rollout_steps,
        "num_epochs": 1,
        "device": "cpu",
    }
    config = TrainingConfig(**{**defaults, **overrides})
    trainer = PPOTrainer(config)
    hx_initial = trainer.network.init_hx(config.num_envs, trainer.device)
    _populate_buffer_with_random_rollout(trainer.buffer, hx_initial)
    return trainer


def _populate_buffer_with_random_rollout(buffer, hx_initial, seed: int = 0) -> None:
    """Fill a RolloutBuffer with synthetic rollout data and run advantages."""
    rng = np.random.default_rng(seed)
    T, N = buffer.rollout_steps, buffer.num_envs
    buffer.obs_cards[:] = rng.integers(0, 256, buffer.obs_cards.shape, dtype=np.uint8)
    buffer.obs_global[:] = rng.integers(0, 256, buffer.obs_global.shape, dtype=np.uint8)
    buffer.obs_actions[:] = rng.integers(0, 256, buffer.obs_actions.shape, dtype=np.uint8)
    buffer.obs_mask[:] = 1
    buffer.actions[:] = rng.integers(0, 32, (T, N), dtype=np.int64)
    buffer.log_probs[:] = rng.standard_normal((T, N)).astype(np.float32)
    buffer.rewards[:] = rng.standard_normal((T, N)).astype(np.float32)
    buffer.dones[:] = 0.0
    buffer.values[:] = rng.standard_normal((T, N)).astype(np.float32)
    buffer.advantages[:] = rng.standard_normal((T, N)).astype(np.float32)
    buffer.returns[:] = buffer.advantages + buffer.values
    buffer.hx_initial = hx_initial
    buffer._ptr = T


def test_recurrent_minibatch_shape_and_count():
    """Plan test #4 (shape half).  With T=32, L=8, num_envs=4,
    minibatch_size=32: envs_per_minibatch=1, four minibatches per epoch,
    each shape (T=32, env_mb=1, ...)."""
    from yugioh_rl.ppo import RolloutBuffer

    buffer = RolloutBuffer(rollout_steps=32, num_envs=4)
    hx_initial = (torch.zeros(1, 4, 64), torch.zeros(1, 4, 64))
    _populate_buffer_with_random_rollout(buffer, hx_initial)

    batches = list(buffer.get_recurrent_batches(minibatch_size=32, device=torch.device("cpu")))
    assert len(batches) == 4
    for b in batches:
        assert b.obs_cards.shape == (32, 1, MAX_CARDS, CARD_FEATURES)
        assert b.obs_global.shape == (32, 1, GLOBAL_FEATURES)
        assert b.actions.shape == (32, 1)
        assert b.dones.shape == (32, 1)
        h, c = b.hx_initial
        assert h.shape == (1, 1, 64)
        assert c.shape == (1, 1, 64)


def test_recurrent_chunk_walk_calls_forward_T_over_L_times(monkeypatch):
    """Plan test #4 (chunk-walk half).  The TBPTT update should call
    network.forward exactly T/L times per minibatch, each with the
    chunk's seq_shape and dones."""
    trainer = _make_tbptt_trainer(rollout_steps=32, num_envs=4)

    seen: list[tuple[int, tuple[int, int]]] = []
    real_forward = trainer.network.forward

    def spy(*args, **kwargs):
        seen.append((args[0].shape[0], kwargs.get("seq_shape")))
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(trainer.network, "forward", spy)
    trainer._run_update_tbptt()

    # 4 minibatches per epoch × T/L = 4 chunks per minibatch = 16 calls
    assert len(seen) == 16
    for batch_dim, seq_shape in seen:
        assert seq_shape == (8, 1)
        assert batch_dim == 8 * 1  # L * env_mb


def test_tbptt_per_chunk_backward_matches_single_backward():
    """Plan test #10 (aggregation half).  Per-chunk weighted backward — the
    memory-bounded TBPTT path — must accumulate into the same total
    gradient as a single .backward() of .mean() over T*env_mb samples.

    Guards the two scaling bugs the plan called out:
    - Per-chunk .mean() summed without scale → grad scales by T/L.
    - Per-chunk .sum() summed → grad scales by T*env_mb.
    """
    torch.manual_seed(0)
    L, env_mb, n_chunks = 8, 2, 4
    T = L * n_chunks
    total = T * env_mb
    scale = L / T

    x = torch.randn(total)

    # Reference: single .mean() backward through a parameter.
    w_ref = torch.tensor(1.0, requires_grad=True)
    (x * w_ref).mean().backward()

    # TBPTT: per-chunk weighted backward, accumulating into w.grad.
    w_tbp = torch.tensor(1.0, requires_grad=True)
    chunks = [x[i * L * env_mb : (i + 1) * L * env_mb] for i in range(n_chunks)]
    for chunk in chunks:
        ((chunk * w_tbp).mean() * scale).backward()

    assert torch.allclose(w_ref.grad, w_tbp.grad)


def test_tbptt_releases_chunk_graph_each_iteration(monkeypatch):
    """The TBPTT loop must call .backward() inside the chunk for-loop so
    each chunk's autograd graph is freed before the next forward.
    Accumulating grad-tracking tensors and only calling .backward() at
    the end keeps every chunk's activations alive simultaneously,
    defeating the TBPTT memory bound.
    """
    trainer = _make_tbptt_trainer(rollout_steps=32, num_envs=4)

    # Record the order of forward / backward calls.  A correct TBPTT loop
    # interleaves them: forward, backward, forward, backward, ...  A wrong
    # one does forward × T/L then backward × 1.
    events: list[str] = []
    real_forward = trainer.network.forward

    def spy_forward(*args, **kwargs):
        events.append("forward")
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(trainer.network, "forward", spy_forward)

    real_backward = torch.Tensor.backward

    def spy_backward(self, *args, **kwargs):
        events.append("backward")
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy_backward)

    trainer._run_update_tbptt()

    # Per minibatch: 4 chunks → expect alternating fwd/bwd (8 events).
    # 4 minibatches per epoch → 32 events total.
    assert len(events) == 32
    for i in range(0, 32, 2):
        assert events[i] == "forward", (
            f"event {i} should be a forward (interleaved per chunk); got: {events[i : i + 2]}"
        )
        assert events[i + 1] == "backward", (
            f"event {i + 1} should be a backward immediately after the chunk "
            f"forward; got: {events[i : i + 2]}"
        )


def test_tbptt_update_changes_network_weights(tmp_path):
    """Plan test #5 (real training half).  Run one TBPTT update and verify
    network parameters actually move — exercises the full chunk loop with
    backward + optimizer.step."""
    trainer = _make_tbptt_trainer(rollout_steps=16, num_envs=2)

    before = {k: v.clone() for k, v in trainer.network.state_dict().items()}
    trainer._run_update_tbptt()
    after = trainer.network.state_dict()

    assert any(
        not torch.allclose(before[k], after[k]) for k in before if before[k].dtype.is_floating_point
    ), "TBPTT update must move at least one parameter"


def test_tbptt_checkpoint_roundtrip_after_training(tmp_path):
    """Plan test #5 (round-trip half).  Train two TBPTT updates, save, reload
    via from_state_dict, forward on the same input — outputs must match."""
    trainer = _make_tbptt_trainer(rollout_steps=16, num_envs=2, num_epochs=2)
    trainer._run_update_tbptt()
    trainer.network.eval()

    cards, glob, actions, mask = _dummy_obs_tensors(batch=2)
    hx_test = trainer.network.init_hx(2, trainer.device)
    with torch.no_grad():
        logits_ref, values_ref, _ = trainer.network(cards, glob, actions, mask, hx=hx_test)

    ckpt_path = str(tmp_path / "trained_rnn.pt")
    torch.save(
        {"config": trainer.config, "model_state_dict": trainer.network.state_dict()},
        ckpt_path,
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded = YuGiOhNet.from_state_dict(blob["config"], blob["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        logits_new, values_new, _ = reloaded(cards, glob, actions, mask, hx=hx_test)

    assert torch.allclose(logits_ref, logits_new)
    assert torch.allclose(values_ref, values_new)


def test_checkpoint_compat_rejects_rnn_type_mismatch(tmp_path):
    """Plan test #2.  Loading a 'none' ckpt with CLI rnn_type='lstm' (or
    vice versa) must raise — RNN cannot be hot-added or hot-removed
    relative to trained weights."""
    from yugioh_rl.ppo import PPOTrainer

    none_ckpt = str(tmp_path / "none.pt")
    _save_minimal_checkpoint(none_ckpt, TrainingConfig())

    cli_with_rnn = TrainingConfig(
        rnn_type="lstm",
        rnn_hidden_dim=64,
        init_checkpoint=none_ckpt,
        save_dir=str(tmp_path / "run1"),
    )
    with pytest.raises(ValueError, match="rnn_type"):
        PPOTrainer(cli_with_rnn)

    rnn_ckpt = str(tmp_path / "lstm.pt")
    _make_rnn_checkpoint(rnn_ckpt, rnn_type="lstm")
    cli_without_rnn = TrainingConfig(
        rnn_hidden_dim=64,  # match ckpt to isolate rnn_type as the only mismatch
        init_checkpoint=rnn_ckpt,
        save_dir=str(tmp_path / "run2"),
    )
    with pytest.raises(ValueError, match="rnn_type"):
        PPOTrainer(cli_without_rnn)


def test_checkpoint_compat_rejects_rnn_hidden_dim_mismatch(tmp_path):
    """When both sides instantiate an RNN, hidden-dim mismatch must reject."""
    from yugioh_rl.ppo import PPOTrainer

    ckpt_path = str(tmp_path / "lstm64.pt")
    _make_rnn_checkpoint(ckpt_path, rnn_type="lstm")  # rnn_hidden_dim=64 from helper

    cli = TrainingConfig(
        rnn_type="lstm",
        rnn_hidden_dim=128,
        init_checkpoint=ckpt_path,
        save_dir=str(tmp_path / "run"),
    )
    with pytest.raises(ValueError, match="rnn_hidden_dim"):
        PPOTrainer(cli)


def test_checkpoint_compat_ignores_rnn_dims_when_both_feed_forward(tmp_path):
    """Feed-forward ckpts (rnn_type='none' on both sides) must load even
    when rnn_hidden_dim / rnn_num_layers placeholder values disagree —
    those fields don't shape any tensor in feed-forward mode.
    """
    from yugioh_rl.ppo import PPOTrainer

    ckpt_config = TrainingConfig(rnn_type="none", rnn_hidden_dim=512, rnn_num_layers=3)
    ckpt_path = str(tmp_path / "ff_with_drift.pt")
    _save_minimal_checkpoint(ckpt_path, ckpt_config)

    # CLI carries today's defaults (256 / 1) — different from the saved ckpt.
    cli = TrainingConfig(
        rnn_type="none",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_checkpoint=ckpt_path,
        save_dir=str(tmp_path / "run"),
    )
    PPOTrainer(cli)  # should not raise


def test_validate_effective_config_tbptt_invariants(monkeypatch, capsys, tmp_path):
    """Plan test #11.  Each TBPTT invariant should fail with a useful
    message when violated; the default RNN combination should pass."""
    from cli.train import validate_effective_config

    # validate_deck_paths needs a real-looking file; use the real one.
    real_deck = "assets/decks/blue_eyes.ydk"

    def cfg(**overrides):
        return TrainingConfig(
            deck_paths=[real_deck],
            rnn_type="lstm",
            rnn_hidden_dim=64,
            **overrides,
        )

    cases = [
        # (overrides, expected substring in stderr)
        (dict(rollout_steps=10, bptt_chunk_len=3, minibatch_size=10, num_envs=8), "bptt-chunk-len"),
        (
            dict(rollout_steps=16, bptt_chunk_len=8, minibatch_size=8, num_envs=8),
            ">= --rollout-steps",
        ),
        (
            dict(rollout_steps=16, bptt_chunk_len=8, minibatch_size=24, num_envs=8),
            "must be divisible by --rollout-steps",
        ),
        (
            dict(rollout_steps=16, bptt_chunk_len=8, minibatch_size=128, num_envs=4),
            "fewer total samples",
        ),
        (
            dict(rollout_steps=16, bptt_chunk_len=8, minibatch_size=48, num_envs=8),
            "envs_per_minibatch",
        ),
    ]
    for overrides, expected in cases:
        with pytest.raises(SystemExit):
            validate_effective_config(cfg(**overrides))
        err = capsys.readouterr().err
        assert expected in err, f"{overrides} did not produce '{expected}' in stderr; got: {err}"

    # Default RNN combination should pass cleanly.
    validate_effective_config(
        cfg(
            rollout_steps=256,
            bptt_chunk_len=16,
            minibatch_size=256,
            num_envs=8,
        )
    )


def test_validate_effective_config_skips_tbptt_invariants_when_rnn_none(tmp_path):
    """rnn_type='none' must skip the TBPTT checks entirely so existing
    feed-forward configs with non-divisible minibatch_size keep working."""
    from cli.train import validate_effective_config

    config = TrainingConfig(
        deck_paths=["assets/decks/blue_eyes.ydk"],
        rnn_type="none",
        # These would all fail TBPTT invariants but should be ignored at "none".
        rollout_steps=256,
        bptt_chunk_len=7,
        minibatch_size=200,
        num_envs=8,
    )
    validate_effective_config(config)


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


def test_ppo_trainer_rejects_mps_lstm_combo():
    """PyTorch 2.11 MPS LSTM backward kernel asserts during the per-step
    TBPTT replay (uint32 underflow in MPSNDArrayDescriptor — see
    bugs/mps_lstm_per_step_backward/mps_lstm_minimal.py). Trainer should fail fast so
    a multi-hour run doesn't crash with a cryptic Metal driver assertion.
    The guard checks ``device.type``, which works regardless of whether MPS
    hardware is actually available — `torch.device("mps")` is just a label.
    """
    from yugioh_rl.ppo import PPOTrainer

    cfg = TrainingConfig(
        num_envs=2,
        deck_paths=["assets/decks/blue_eyes.ydk"],
        rnn_type="lstm",
        device="mps",
    )
    with pytest.raises(RuntimeError, match="rnn_type='lstm' on device='mps'"):
        PPOTrainer(cfg)


def test_ppo_trainer_allows_mps_gru_and_mps_none(tmp_path):
    """Companion to the LSTM+MPS guard: GRU and rnn_type='none' on MPS must
    still construct.  Build via ``init_checkpoint`` so the trainer never
    materializes an actual MPS tensor (CI may not have MPS hardware), only
    exercises the device-type branch in __init__.
    """
    from yugioh_rl.ppo import PPOTrainer

    # Pre-make a tiny CPU checkpoint so PPOTrainer can run __init__ without
    # actually allocating on MPS — `init_checkpoint` loads with map_location
    # set to self.device, but if MPS isn't present this would still fail.
    # Skip if MPS isn't available at all so the test runs only where it can.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    for rnn_type in ("gru", "none"):
        cfg = TrainingConfig(
            num_envs=2,
            deck_paths=["assets/decks/blue_eyes.ydk"],
            rnn_type=rnn_type,
            device="mps",
        )
        # Should not raise.
        PPOTrainer(cfg)
