"""Actor-learner vec env (workers hold local agent policy).

In contrast to ``SubprocVecEnv``, each worker process holds a private copy of
the agent policy and runs inference locally — eliminating the per-step
trainer↔worker pipe round-trip. The trainer publishes new weights to shared
memory; workers refresh from shared memory at rollout boundaries.

Two modes: ``sync_actor_learner`` (barrier — trainer waits for all N rollouts)
and ``async_actor_learner`` (no barrier — workers push to a queue, trainer
drains K qualifying rollouts per update, V-trace corrects for staleness).
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from yugioh_rl.policy_inputs import build_forward_inputs

if TYPE_CHECKING:
    import torch.nn as nn

    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.env_wrapper import DeckDict
    from yugioh_rl.network import HxState
    from yugioh_rl.opponent_pool import Sampling


__all__ = [
    "ActorLearnerVecEnv",
    "AsyncActorLearnerVecEnv",
    "WorkerDiedError",
    "WorkerTimeoutError",
]


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
    emitted as the ``final_obs_*`` keys (single per-env arrays, not
    stacked). ``infos`` is the list of per-step info dicts from env.step()
    so the trainer can drive episode tracking the same way it does in
    SubprocVecEnv (terminal_reward / steps / agent_deck_idx).

    ``final_hx`` is the post-rollout hidden state (after mask_hx on the last
    done). It is None for feed-forward configs, a single tensor of shape
    ``(num_layers, 1, hidden_dim)`` for GRU, or a tuple of two such tensors
    for LSTM. The trainer stacks per-env ``final_hx`` for the GAE bootstrap.
    """
    obs_cards = np.stack([t.obs["cards"] for t in transitions])
    obs_global = np.stack([t.obs["global_state"] for t in transitions])
    obs_actions = np.stack([t.obs["actions"] for t in transitions])
    action_mask = np.stack([t.obs["action_mask"] for t in transitions])
    obs_chain = np.stack([t.obs["pending_chain"] for t in transitions])
    obs_event = np.stack([t.obs["event_history"] for t in transitions])
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
        "obs_chain": obs_chain,
        "obs_event": obs_event,
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
        "final_obs_chain": final_obs["pending_chain"],
        "final_obs_event": final_obs["event_history"],
        "infos": [t.info for t in transitions],
        "final_hx": final_hx,
    }


def _init_worker(env_kwargs, weight_handles, config_dict):
    """Shared worker-process setup: imports, thread limits, policy, env.

    Returns ``(weights, local_policy, env, obs)`` — everything a worker
    needs before entering its rollout loop.  Deferred imports keep torch
    out of the parent process.
    """
    import torch

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
    return weights, local_policy, env, obs


def _collect_one_rollout(
    local_policy, env, obs, hx, version, rollout_steps, done_t, shutdown_check=None
):
    """Run T inference+env-step iterations, return (transitions, obs, hx).

    ``done_t`` is a pre-allocated single-element float tensor for mask_hx.
    If ``shutdown_check`` is provided (a callable returning bool), the
    function returns early with ``None`` when it fires.

    Both sync and async workers call this for the inner rollout loop.
    """
    import torch
    from torch.distributions import Categorical

    transitions: list[Transition] = []
    for _ in range(rollout_steps):
        if shutdown_check is not None and shutdown_check():
            return None

        # torch.from_numpy aliases obs's numpy buffers; safe because
        # TrainingEnv.step() returns fresh numpy arrays each call,
        # so the obs reference held in `transitions` is never
        # mutated underneath us.
        with torch.no_grad():
            inputs = build_forward_inputs(obs, add_batch_dim=True)
            logits, value, hx_new = local_policy(**inputs, hx=hx)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        a_int = int(action.item())
        next_obs, reward, done, info = env.step(a_int)
        transitions.append(
            Transition(
                obs=obs,
                action=a_int,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=float(reward),
                done=bool(done),
                version=int(version),
                info=info,
            )
        )
        done_t[0] = float(done)
        hx = local_policy.mask_hx(hx_new, done_t)
        # Explicit reset on done: step() returns the terminal obs
        # (no auto-reset), and the next iteration would otherwise
        # feed a finished duel's obs back into the policy.
        if done:
            next_obs = env.reset()
        obs = next_obs

    return transitions, obs, hx


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
    """
    import torch

    weights, local_policy, env, obs = _init_worker(env_kwargs, weight_handles, config_dict)
    done_t = torch.zeros(1, dtype=torch.float32)

    try:
        while True:
            cmd, _ = remote.recv()
            if cmd == "shutdown":
                break
            assert cmd == "go", f"unexpected cmd {cmd!r}"

            version = weights.refresh_into(local_policy)
            # Reset hx every rollout: post-update weights make stale hx inconsistent.
            hx = local_policy.init_hx(1, "cpu")
            transitions, obs, hx = _collect_one_rollout(
                local_policy,
                env,
                obs,
                hx,
                version,
                rollout_steps,
                done_t,
            )

            remote.send(
                (
                    "rollout",
                    _pack_rollout(
                        transitions,
                        final_obs=obs,
                        final_hx=hx,
                    ),
                )
            )
    finally:
        env.close()


def _async_actor_learner_worker(
    queue,
    shutdown_event,
    env_kwargs: dict,
    weight_handles: dict,
    config_dict: dict,
    rollout_steps: int,
) -> None:
    """Async worker: continuously produces rollouts pushed to *queue*.

    Runs until ``shutdown_event`` is set. Refreshes weights from shared
    memory at each rollout boundary only when the trainer has published a
    new version; otherwise reuses the current weights and carries hx forward.
    """
    import torch

    weights, local_policy, env, obs = _init_worker(env_kwargs, weight_handles, config_dict)
    done_t = torch.zeros(1, dtype=torch.float32)

    current_version = weights.version
    hx = local_policy.init_hx(1, "cpu")

    try:
        while not shutdown_event.is_set():
            # Check for new weights at rollout boundary
            latest_version = weights.version
            if latest_version != current_version:
                current_version = weights.refresh_into(local_policy)
                hx = local_policy.init_hx(1, "cpu")

            result = _collect_one_rollout(
                local_policy,
                env,
                obs,
                hx,
                current_version,
                rollout_steps,
                done_t,
                shutdown_check=shutdown_event.is_set,
            )
            if result is None:
                return
            transitions, obs, hx = result

            queue.put(_pack_rollout(transitions, final_obs=obs, final_hx=hx))
    finally:
        env.close()


class _BaseActorLearnerVecEnv:
    """Shared initialization for sync and async actor-learner vec envs.

    Subclasses override ``_spawn_workers`` to set up the IPC mechanism
    (pipe for sync, queue+event for async) and ``close`` to tear down.
    """

    def __init__(
        self,
        num_envs: int,
        deck_pool: list[DeckDict],
        opponent: str,
        reward_shaping: bool,
        shaping_lp_weight: float,
        shaping_card_weight: float,
        seed: int,
        agent_player: str,
        opponent_device: str | None,
        master_model: nn.Module,
        config: TrainingConfig,
        rollout_steps: int,
        deck_allocation: str = "random",
        mirror_decks: bool = False,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_sampling: Sampling = "uniform",
        opponent_pool_config: TrainingConfig | None = None,
        max_steps: int = 2000,
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
            "deck_allocation": deck_allocation,
            "mirror_decks": mirror_decks,
            "opponent_device": opponent_device,
            "opponent_pool_handles": opponent_pool_handles,
            "opponent_pool_temperature": opponent_pool_temperature,
            "opponent_pool_sampling": opponent_pool_sampling,
            "opponent_pool_config": opponent_pool_config,
            "max_steps": max_steps,
        }

        ctx = mp.get_context("spawn")
        from yugioh_rl.env_wrapper import limit_worker_blas_threads

        limit_worker_blas_threads()

        self._workers: list[mp.Process] = []
        self._spawn_workers(ctx, base_env_kwargs, seed, weight_handles, config_dict, rollout_steps)

    def _spawn_workers(
        self, ctx, base_env_kwargs, seed, weight_handles, config_dict, rollout_steps
    ):
        raise NotImplementedError

    def publish_weights(self, model) -> int:
        """Trainer-side: write fresh weights to shared memory."""
        return self.shared_weights.publish(model)

    def _reap_workers(self) -> None:
        for proc in self._workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._reap_workers()


class AsyncActorLearnerVecEnv(_BaseActorLearnerVecEnv):
    """Async vec env: workers run continuously, push rollouts to a queue.

    ``collect_rollouts(max_version_lag)`` drains the queue until K qualifying
    rollouts are collected (where K = ``num_envs``), discarding any whose
    version lag exceeds ``max_version_lag``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trainer_version = self.shared_weights.version

    def _spawn_workers(
        self, ctx, base_env_kwargs, seed, weight_handles, config_dict, rollout_steps
    ):
        self._queue = ctx.Queue()
        self._shutdown = ctx.Event()
        for i in range(self.num_envs):
            p = ctx.Process(
                target=_async_actor_learner_worker,
                kwargs={
                    "queue": self._queue,
                    "shutdown_event": self._shutdown,
                    "env_kwargs": {**base_env_kwargs, "seed": seed + i * 10000},
                    "weight_handles": weight_handles,
                    "config_dict": config_dict,
                    "rollout_steps": rollout_steps,
                },
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    @property
    def trainer_version(self) -> int:
        return self._trainer_version

    @property
    def queue_depth(self) -> int | None:
        """Current queue size, or None if the platform cannot report it."""
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return None

    def collect_rollouts(self, max_version_lag: int) -> tuple[list[dict], int, list[int]]:
        """Drain K qualifying rollouts from the queue.

        Keeps pulling until ``num_envs`` rollouts with version lag
        <= ``max_version_lag`` are collected. Returns
        ``(rollouts, discarded_count, version_lags)``.

        Raises ``WorkerDiedError`` if all workers have died and the queue
        is empty before K rollouts are collected.
        """
        import queue as queue_mod

        rollouts: list[dict] = []
        version_lags: list[int] = []
        discarded = 0

        while len(rollouts) < self.num_envs:
            try:
                payload = self._queue.get(timeout=self._worker_timeout_s)
            except queue_mod.Empty:
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    raise WorkerDiedError("all async actor-learner workers have died") from None
                raise WorkerTimeoutError(
                    f"no rollout received for {self._worker_timeout_s}s "
                    f"({alive}/{self.num_envs} workers alive)"
                ) from None

            pv = payload["policy_version"]
            version = int(pv) if isinstance(pv, int) else int(pv[0])
            lag = self._trainer_version - version
            if lag <= max_version_lag:
                assert payload["actions"].shape == (self.rollout_steps,), (
                    f"rollout shape {payload['actions'].shape} mismatches "
                    f"expected ({self.rollout_steps},)"
                )
                rollouts.append(payload)
                version_lags.append(lag)
            else:
                discarded += 1

        return rollouts, discarded, version_lags

    def publish_weights(self, model) -> int:
        """Publish weights and update trainer version."""
        v = super().publish_weights(model)
        self._trainer_version = v
        return v

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        self._reap_workers()


class WorkerDiedError(RuntimeError):
    """Raised by ActorLearnerVecEnv when a worker exits unexpectedly."""


class WorkerTimeoutError(RuntimeError):
    """Raised when a worker is alive but has not delivered a rollout within
    ``worker_timeout_s``. Distinguishes a stuck-but-alive worker (e.g. env
    deadlock) from one that crashed (``WorkerDiedError``)."""


class ActorLearnerVecEnv(_BaseActorLearnerVecEnv):
    """Vec env where each worker holds a local agent policy.

    Replaces SubprocVecEnv when ``vec_env_type == "sync_actor_learner"``.
    Per cycle: ``publish_weights`` then ``collect_rollouts`` (which sends
    ``go`` to all workers and blocks on N rollouts).
    """

    def _spawn_workers(
        self, ctx, base_env_kwargs, seed, weight_handles, config_dict, rollout_steps
    ):
        self._remotes = []
        for i in range(self.num_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_actor_learner_worker,
                kwargs={
                    "remote": child_conn,
                    "env_kwargs": {**base_env_kwargs, "seed": seed + i * 10000},
                    "weight_handles": weight_handles,
                    "config_dict": config_dict,
                    "rollout_steps": rollout_steps,
                },
                daemon=True,
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
        for remote, proc in zip(self._remotes, self._workers, strict=True):
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            with suppress(BrokenPipeError, EOFError):
                remote.send(("shutdown", None))
        self._reap_workers()
