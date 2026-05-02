"""Actor-learner vec env (workers hold local agent policy).

In contrast to ``SubprocVecEnv``, each worker process holds a private copy of
the agent policy and runs inference locally — eliminating the per-step
trainer↔worker pipe round-trip. The trainer publishes new weights to shared
memory at rollout boundaries; workers refresh from shared memory at the
start of each rollout.

Currently sync-only. The ``policy_version`` tag on each transition is the
async-readiness hook (uniform-per-rollout in sync; would vary per-step in
async). Async-mode worker control flow lands when async support does.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


__all__ = ["Transition", "_pack_rollout", "_actor_learner_worker"]


class Transition(NamedTuple):
    """One step of a worker-side rollout. Attributes match the rollout
    payload schema; adding a field here (e.g. ``hx`` for RNN) ripples
    cleanly through ``_pack_rollout`` instead of breaking positional reads.
    """
    obs: dict[str, np.ndarray]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    version: int


def _pack_rollout(transitions: list[Transition]) -> dict:
    """Stack per-step records into a dict of numpy arrays.

    In sync mode the version is uniform across the rollout and collapses to a
    scalar int; in async it remains a per-step int64 array.
    """
    obs_cards = np.stack([t.obs["cards"] for t in transitions])
    obs_global = np.stack([t.obs["global_state"] for t in transitions])
    obs_actions = np.stack([t.obs["actions"] for t in transitions])
    action_mask = np.stack([t.obs["action_mask"] for t in transitions])
    actions = np.array([t.action for t in transitions], dtype=np.int64)
    log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)
    values = np.array([t.value for t in transitions], dtype=np.float32)
    rewards = np.array([t.reward for t in transitions], dtype=np.float32)
    dones = np.array([t.done for t in transitions], dtype=bool)
    versions = np.array([t.version for t in transitions], dtype=np.int64)
    if versions.min() == versions.max():
        policy_version: Any = int(versions[0])
    else:
        policy_version = versions
    return {
        "obs_cards": obs_cards,
        "obs_global": obs_global,
        "obs_actions": obs_actions,
        "action_mask": action_mask,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "policy_version": policy_version,
    }


def _actor_learner_worker(
    remote,
    env_kwargs: dict,
    weight_handles: dict,
    config_dict: dict,
    rollout_steps: int,
) -> None:
    """Worker process: own a TrainingEnv + a local policy, drive a rollout loop.

    Sync protocol: blocks on ``remote.recv()`` for ``("go", v)``; refreshes
    from shared memory; runs T inference+env-step iterations; sends one
    ``("rollout", payload)``. Shutdown via ``("shutdown", None)``.

    The local policy's parameter shapes come from the shared weight tensors
    (which the trainer ``publish``-es before spawning workers), so no
    state-dict pickle is needed and ``card_text_embeddings.pt`` is never
    re-loaded in worker processes.
    """
    import torch
    from torch.distributions import Categorical

    from yugioh_rl.config import TrainingConfig, normalize_legacy_config
    from yugioh_rl.env_wrapper import TrainingEnv
    from yugioh_rl.network import YuGiOhNet
    from yugioh_rl.shared_weights import SharedPolicyWeights

    cfg = TrainingConfig(**config_dict)
    cfg = normalize_legacy_config(cfg)
    if cfg.is_recurrent:
        # Worker does not yet thread hx between steps; trainer's existing
        # SubprocVecEnv path does this via mask_hx. Until added, an RNN
        # config would silently produce zero-hidden-state rollouts.
        raise NotImplementedError(
            "actor-learner worker does not yet support rnn_type != 'none'"
        )

    # Pin per-worker policy RNG so action sampling is deterministic given
    # env_kwargs['seed']. The duel and deck RNGs are seeded inside TrainingEnv.
    torch.manual_seed(int(env_kwargs.get("seed", 0)))

    weights = SharedPolicyWeights.from_handles(weight_handles)
    # The shared tensors hold the trainer's published initial weights —
    # use them directly as the state_dict. from_state_dict consumes shape
    # info plus values; refresh_into below would be a no-op so we skip it.
    local_policy = YuGiOhNet.from_state_dict(cfg, weight_handles["tensors"])
    local_policy.eval()

    env = TrainingEnv(**env_kwargs)
    obs = env.reset()

    try:
        while True:
            cmd, _ = remote.recv()
            if cmd == "shutdown":
                break
            assert cmd == "go", f"unexpected cmd {cmd!r}"

            version = weights.refresh_into(local_policy)
            transitions: list[Transition] = []
            for _ in range(rollout_steps):
                # torch.from_numpy aliases obs's numpy buffers; safe because
                # TrainingEnv.step() returns fresh numpy arrays each call,
                # so the obs reference held in `transitions` is never
                # mutated underneath us.
                with torch.no_grad():
                    cards_t = torch.from_numpy(obs["cards"]).unsqueeze(0)
                    glob_t = torch.from_numpy(obs["global_state"]).unsqueeze(0)
                    acts_t = torch.from_numpy(obs["actions"]).unsqueeze(0)
                    mask_t = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                    logits, value, _ = local_policy(cards_t, glob_t, acts_t, mask_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                a_int = int(action.item())
                next_obs, reward, done, info = env.step(a_int)
                transitions.append(Transition(
                    obs=obs,
                    action=a_int,
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                    reward=float(reward),
                    done=bool(done),
                    version=int(version),
                ))
                obs = next_obs

            remote.send(("rollout", _pack_rollout(transitions)))
    finally:
        env.close()
