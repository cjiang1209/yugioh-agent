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

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import DeckDict
    from yugioh_rl.network import HxState
    from yugioh_rl.opponent_pool import Sampling


__all__ = ["Transition", "_pack_rollout", "_actor_learner_worker",
           "ActorLearnerVecEnv", "WorkerDiedError", "WorkerTimeoutError"]


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
    info: dict


def _pack_rollout(
    transitions: list[Transition],
    final_obs: dict[str, np.ndarray],
    final_hx: HxState,
) -> dict:
    """Stack per-step records into a dict of numpy arrays.

    In sync mode the version is uniform across the rollout and collapses to a
    scalar int; in async it remains a per-step int64 array.

    ``final_obs`` is the post-rollout observation (after the last env.step).
    The trainer uses it to bootstrap the value estimate for GAE; it is
    emitted as the four ``final_obs_*`` keys (single per-env arrays, not
    stacked). ``infos`` is the list of per-step info dicts from env.step()
    so the trainer can drive episode tracking the same way it does in
    SubprocVecEnv (terminal_reward / episode_length / agent_deck_idx).

    ``final_hx`` is the post-rollout hidden state (after mask_hx on the last
    done). It is None for feed-forward configs, a single tensor of shape
    ``(num_layers, 1, hidden_dim)`` for GRU, or a tuple of two such tensors
    for LSTM. The trainer stacks per-env ``final_hx`` for the GAE bootstrap.
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
        "final_obs_cards": final_obs["cards"],
        "final_obs_global": final_obs["global_state"],
        "final_obs_actions": final_obs["actions"],
        "final_action_mask": final_obs["action_mask"],
        "infos": [t.info for t in transitions],
        "final_hx": final_hx,
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

    # batch_size=1 inference — multi-threaded BLAS only adds scheduling overhead.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    cfg = TrainingConfig(**config_dict)
    cfg = normalize_legacy_config(cfg)

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
            # Reset hx every rollout: post-update weights make stale hx inconsistent.
            hx = local_policy.init_hx(1, "cpu")
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
                    logits, value, hx_new = local_policy(
                        cards_t, glob_t, acts_t, mask_t, hx=hx,
                    )
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
                    info=info,
                ))
                done_t = torch.tensor([float(done)], dtype=torch.float32)
                hx = local_policy.mask_hx(hx_new, done_t)
                # Explicit reset on done: step() returns the terminal obs
                # (no auto-reset), and the next iteration would otherwise
                # feed a finished duel's obs back into the policy.
                if done:
                    next_obs = env.reset()
                obs = next_obs

            remote.send(("rollout", _pack_rollout(
                transitions, final_obs=obs, final_hx=hx,
            )))
    finally:
        env.close()


class WorkerDiedError(RuntimeError):
    """Raised by ActorLearnerVecEnv when a worker exits unexpectedly."""


class WorkerTimeoutError(RuntimeError):
    """Raised when a worker is alive but has not delivered a rollout within
    ``worker_timeout_s``. Distinguishes a stuck-but-alive worker (e.g. env
    deadlock) from one that crashed (``WorkerDiedError``)."""


class ActorLearnerVecEnv:
    """Vec env where each worker holds a local agent policy.

    Replaces SubprocVecEnv when ``vec_env_type == "sync_actor_learner"``.
    Per cycle: ``publish_weights`` then ``collect_rollouts`` (which sends
    ``go`` to all workers and blocks on N rollouts).
    """

    def __init__(
        self,
        num_envs: int,
        deck_pool: list["DeckDict"],
        opponent: str,
        reward_shaping: bool,
        shaping_lp_weight: float,
        shaping_card_weight: float,
        seed: int,
        agent_player: str,
        opponent_device: str | None,
        master_model: "nn.Module",
        config: "TrainingConfig",
        rollout_steps: int,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_sampling: "Sampling" = "uniform",
        opponent_pool_config: "TrainingConfig | None" = None,
        worker_timeout_s: float = 300.0,
    ) -> None:
        import multiprocessing as mp
        from dataclasses import asdict

        from yugioh_rl.shared_weights import SharedPolicyWeights

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self._worker_timeout_s = worker_timeout_s
        self._closed = False

        # Trainer publishes initial weights BEFORE spawning workers so each
        # worker's first read of shared memory sees a populated buffer.
        self.shared_weights = SharedPolicyWeights(master_model)
        self.shared_weights.publish(master_model)
        config_dict = asdict(config)
        weight_handles = self.shared_weights.share_handles()
        base_env_kwargs = {
            "deck_pool": deck_pool,
            "opponent": opponent,
            "reward_shaping": reward_shaping,
            "shaping_lp_weight": shaping_lp_weight,
            "shaping_card_weight": shaping_card_weight,
            "agent_player": agent_player,
            "opponent_device": opponent_device,
            "opponent_pool_handles": opponent_pool_handles,
            "opponent_pool_temperature": opponent_pool_temperature,
            "opponent_pool_sampling": opponent_pool_sampling,
            "opponent_pool_config": opponent_pool_config,
        }

        ctx = mp.get_context("spawn")
        from yugioh_rl.env_wrapper import limit_worker_blas_threads
        limit_worker_blas_threads()
        self._remotes = []
        self._workers = []
        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            spawn_kwargs = {
                "remote": child_conn,
                "env_kwargs": {**base_env_kwargs, "seed": seed + i * 10000},
                "weight_handles": weight_handles,
                "config_dict": config_dict,
                "rollout_steps": rollout_steps,
            }
            p = ctx.Process(
                target=_actor_learner_worker, kwargs=spawn_kwargs, daemon=True,
            )
            p.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._workers.append(p)

    def collect_rollouts(self) -> list[dict]:
        """Send "go" to all workers and block until all N rollouts arrive.

        Raises ``WorkerDiedError`` if any worker has exited, or
        ``WorkerTimeoutError`` if a worker is alive but silent past the
        configured timeout (so a deadlocked worker can never hang the trainer
        indefinitely).
        """
        version = self.shared_weights.version
        for remote in self._remotes:
            remote.send(("go", version))

        rollouts: list[dict] = []
        for remote, proc in zip(self._remotes, self._workers):
            if not remote.poll(self._worker_timeout_s):
                if not proc.is_alive():
                    raise WorkerDiedError(
                        f"actor-learner worker pid={proc.pid} died "
                        f"(exitcode={proc.exitcode}) before sending rollout"
                    )
                raise WorkerTimeoutError(
                    f"worker pid={proc.pid} alive but silent for "
                    f"{self._worker_timeout_s}s — likely deadlocked"
                )
            cmd, payload = remote.recv()
            assert cmd == "rollout", f"unexpected cmd {cmd!r}"
            assert payload["actions"].shape == (self.rollout_steps,), (
                f"rollout shape {payload['actions'].shape} mismatches "
                f"expected ({self.rollout_steps},)"
            )
            rollouts.append(payload)
        return rollouts

    def publish_weights(self, model) -> int:
        """Trainer-side: write fresh weights to shared memory."""
        return self.shared_weights.publish(model)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("shutdown", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
