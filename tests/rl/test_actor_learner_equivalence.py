"""A/B regression: sync_actor_learner vs subproc PPO with identical config.

The two paths are not bit-equivalent on real workloads — floating-point
non-associativity (different reduction orders across processes), worker-side
RNG that runs independently from the trainer's batched RNG, and the
trainer's batched single-forward-pass-vs-N-single-batch-forward-passes split
all introduce divergence. We instead pin distributional equivalence:
parameter L2 norms agree within 20% after a short run with the same seed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.ppo import PPOTrainer

from tests.rl.conftest import requires_engine


def _param_l2(net: torch.nn.Module) -> float:
    return float(sum(p.detach().pow(2).sum().item() for p in net.parameters()))


@requires_engine
def test_actor_learner_matches_subproc_within_tolerance(tmp_path) -> None:
    deck = "assets/decks/starter.ydk"
    if not Path(deck).exists():
        pytest.skip(f"{deck} not present")

    base = dict(
        num_envs=2,
        deck_paths=[deck],
        opponent="random",
        reward_shaping=False,
        rollout_steps=16,
        num_epochs=2,
        minibatch_size=16,
        total_timesteps=512,         # 2 envs × 16 rollout × 16 updates
        eval_interval=9999,
        save_interval=9999,
        log_interval=999,
        device="cpu",
    )

    cfg_sub = TrainingConfig(
        **base, vec_env_type="subproc", save_dir=str(tmp_path / "subproc"),
    )
    cfg_al = TrainingConfig(
        **base, vec_env_type="sync_actor_learner",
        save_dir=str(tmp_path / "actor_learner"),
    )

    t_sub = PPOTrainer(cfg_sub)
    t_sub.train()
    t_al = PPOTrainer(cfg_al)
    t_al.train()

    n_sub = _param_l2(t_sub.network)
    n_al = _param_l2(t_al.network)
    rel = abs(n_sub - n_al) / max(n_sub, 1e-9)
    assert rel < 0.20, (
        f"param L2 diverged too far: subproc={n_sub:.4f} "
        f"actor_learner={n_al:.4f} rel={rel:.3f}"
    )

    for net, label in [(t_sub.network, "subproc"), (t_al.network, "actor_learner")]:
        for name, p in net.state_dict().items():
            if p.dtype.is_floating_point:
                assert torch.isfinite(p).all(), f"{label}: non-finite param {name}"
