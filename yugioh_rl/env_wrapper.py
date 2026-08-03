"""Training environment wrappers: single-env and vectorized."""

from __future__ import annotations

import multiprocessing as mp
import os
import random as stdlib_random
from typing import TYPE_CHECKING, Any

import numpy as np

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
    decode_u16,
)
from yugioh_env.models import YuGiOhAction
from yugioh_rl.policy_inputs import build_forward_inputs

if TYPE_CHECKING:
    import torch.nn as nn

    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import HxState
    from yugioh_rl.opponent_pool import Sampling

DeckDict = dict[str, list[int]]  # {"main": [int, ...], "extra": [int, ...]}


def parse_deck_pool(deck_paths: list[str]) -> list[DeckDict]:
    """Pre-parse .ydk files into dicts for passing to worker processes.

    Called once in the main process; the resulting list is picklable and
    can be sent to TrainingEnv workers without further file I/O.
    """
    from yugioh_env.deck_parser import parse_ydk

    return [parse_ydk(p) for p in deck_paths]


class TrainingEnv:
    """Single-env wrapper around YuGiOhEnvironment for training and eval.

    Bypasses HTTP by calling the environment directly in-process.  Provides
    numpy observations, reward shaping, and terminal info on episode end.
    Episodes are addressable by index via ``reset(episode_idx=...)`` so a
    parallel-eval worker can dispatch one specific episode at a time.

    ``step()`` does **not** auto-reset on terminal transitions.  The caller
    must call ``reset()`` explicitly before stepping again — vec-env wrappers
    (e.g. ``SubprocVecEnv.reset_done``) restore today's wire-level
    auto-reset semantics around this contract for training paths.
    """

    def __init__(
        self,
        deck_pool: list[DeckDict],
        opponent: str = "greedy",
        reward_shaping: bool = True,
        shaping_lp_weight: float = 0.01,
        shaping_card_weight: float = 0.005,
        seed: int = 42,
        agent_player: str = "random",
        deck_allocation: str = "random",
        mirror_decks: bool = False,
        opponent_device: str | None = None,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_sampling: Sampling = "uniform",
        opponent_pool_config: TrainingConfig | None = None,
        max_steps: int = 2000,
    ) -> None:
        if not deck_pool:
            raise ValueError("deck_pool must not be empty")
        from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

        # Map config strings to environment values
        agent_player_map = {"first": 0, "second": 1, "random": "random"}
        self._agent_player_setting = agent_player_map.get(agent_player, agent_player)

        env_config: dict[str, Any] = {
            "opponent": opponent,
            "agent_player": self._agent_player_setting,
            "collapse_forced": True,
            "max_steps": max_steps,
        }
        # Absent key lets YUGIOH_OPPONENT_DEVICE win; see _resolve_opponent_device.
        if opponent_device is not None:
            env_config["opponent_device"] = opponent_device

        self._env = YuGiOhEnvironment(config=env_config)
        self._reward_shaping = reward_shaping
        self._lp_weight = shaping_lp_weight
        self._card_weight = shaping_card_weight
        self._seed = seed
        self._episode_count = 0

        # Deck pool and DeckSelector for allocation/mirror.
        from yugioh_rl.deck_selector import DeckSelector

        self._deck_pool = deck_pool
        self._selector = DeckSelector(
            pool_size=len(deck_pool), seed=seed, allocation=deck_allocation, mirror=mirror_decks
        )
        self._player_rng = stdlib_random.Random()
        self._last_agent_deck_idx = -1

        # State for reward shaping
        self._prev_my_lp = 0
        self._prev_opp_lp = 0
        self._prev_advantage = 0

        self._opponent_pool = None
        self._current_opp_slot: int | None = None
        if opponent_pool_handles is not None:
            from yugioh_rl.network import YuGiOhNet
            from yugioh_rl.opponent_pool import OpponentPool

            if opponent_pool_config is None:
                raise ValueError(
                    "opponent_pool_config is required when opponent_pool_handles is set"
                )

            self._opponent_pool = OpponentPool.attach_worker(
                handles=opponent_pool_handles,
                initial_opponent_spec=opponent,
                network_factory=lambda: YuGiOhNet.from_config(opponent_pool_config),
                temperature=opponent_pool_temperature,
                sampling=opponent_pool_sampling,
                rng=stdlib_random.Random(seed + 1),
            )

    def reset(self, *, episode_idx: int | None = None) -> dict[str, np.ndarray]:
        """Begin a new episode and return the first observation.

        By default (``episode_idx=None``) increments the internal episode
        counter — same behavior sequential callers expect (training rollouts,
        run_match). Pass ``episode_idx=N`` to address a specific episode index;
        used by the parallel-eval worker pool to dispatch one specific episode
        at a time without relying on stateful counter advancement.
        """
        if episode_idx is not None:
            self._episode_count = episode_idx
        else:
            self._episode_count += 1
        episode_seed = self._seed + self._episode_count

        # Pre-resolve agent_player so we can map decks to correct engine
        # positions (must match the environment's resolution logic).
        if self._agent_player_setting == "random":
            self._player_rng.seed(episode_seed)
            resolved_player = self._player_rng.randint(0, 1)
        else:
            resolved_player = int(self._agent_player_setting)

        agent_deck_idx, opp_deck_idx = self._selector.select(self._episode_count)
        agent_deck = self._deck_pool[agent_deck_idx]
        opp_deck = self._deck_pool[opp_deck_idx]

        # Map agent/opponent to engine player 0/1
        if resolved_player == 0:
            deck0, deck1 = agent_deck, opp_deck
        else:
            deck0, deck1 = opp_deck, agent_deck

        self._last_agent_deck_idx = agent_deck_idx
        self._last_opp_deck_idx = opp_deck_idx

        if self._opponent_pool is not None:
            self._current_opp_slot, new_opp = self._opponent_pool.sample()
            self._env.set_opponent(new_opp)

        obs = self._env.reset(
            seed=episode_seed,
            deck0=deck0,
            deck1=deck1,
            agent_player=resolved_player,
        )
        np_obs = obs.as_arrays()

        # Initialize shaping state
        gs = np_obs["global_state"]
        self._prev_my_lp = decode_u16(gs, 0)
        self._prev_opp_lp = decode_u16(gs, 2)
        self._prev_advantage = self._compute_advantage(gs)

        return np_obs

    def step(self, action_index: int) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Step the environment with an action.

        On ``done=True`` the returned obs is the **terminal** obs of the
        just-finished episode — there is no auto-reset.  The caller must
        call :meth:`reset` (or, for vec-env wrappers, ``reset_done``)
        before invoking ``step`` again; otherwise the next call would
        advance a finished duel.

        Returns:
            obs: numpy observation dict (terminal on done=True)
            reward: shaped or terminal reward
            done: whether episode ended
            info: extra info — populated on done with ``terminal_reward``,
                  ``steps``, and ``agent_deck_idx``
        """
        obs = self._env.step(YuGiOhAction(action_index=action_index))
        np_obs = obs.as_arrays()
        info: dict[str, Any] = {}

        reward = obs.reward
        done = obs.done

        if done:
            info["terminal_reward"] = reward
            info["steps"] = self._env._step_count
            info["agent_deck_idx"] = self._last_agent_deck_idx
            info["timeout"] = self._env._timed_out
            info["opponent_steps"] = self._env._opp_step_count
            if self._opponent_pool is not None and self._current_opp_slot is not None:
                self._opponent_pool.report_result(
                    slot=self._current_opp_slot,
                    agent_won=reward > 0,
                )
                self._current_opp_slot = None
            # No auto-reset.  Caller chooses when to begin the next episode
            # (vec-env workers do it eagerly to preserve wire behavior;
            # eval workers reset to a specific episode_idx for the next task).
        elif self._reward_shaping:
            reward += self._compute_shaping(np_obs["global_state"])

        return np_obs, reward, done, info

    @staticmethod
    def _compute_advantage(gs: np.ndarray) -> int:
        """Compute card advantage from global state zone counts."""
        # observation.py packs five counts per player after the 2-byte phase:
        # 10=my_deck, 11=my_hand, 12=my_grave, 13=my_banished, 14=my_extra
        # 15=opp_deck, 16=opp_hand, 17=opp_grave, 18=opp_banished, 19=opp_extra
        my_hand = int(gs[11])
        opp_hand = int(gs[16])
        return my_hand - opp_hand

    def _compute_shaping(self, gs: np.ndarray) -> float:
        """Compute reward shaping based on LP and card advantage deltas."""
        my_lp = decode_u16(gs, 0)
        opp_lp = decode_u16(gs, 2)
        advantage = self._compute_advantage(gs)

        delta_my_lp = my_lp - self._prev_my_lp
        delta_opp_lp = opp_lp - self._prev_opp_lp
        delta_advantage = advantage - self._prev_advantage

        shaped = (
            self._lp_weight * (delta_my_lp - delta_opp_lp) / 8000.0
            + self._card_weight * delta_advantage
        )

        self._prev_my_lp = my_lp
        self._prev_opp_lp = opp_lp
        self._prev_advantage = advantage

        return shaped

    @property
    def current_msg(self) -> dict | None:
        return self._env.current_msg

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    def close(self) -> None:
        self._env.close()


class EvalEnv:
    """Lean eval env over YuGiOhEnvironment: DeckSelector + one fixed opponent.

    Exposes the duck-typed env contract the eval loop drives
    (reset/step/current_msg/num_actions/close). Built from picklable kwargs
    so eval workers can construct it in a spawned process.
    """

    def __init__(
        self,
        deck_pool: list[DeckDict],
        opponent: str,
        *,
        seed: int,
        agent_player: str = "random",
        opponent_device: str | None = None,
        deck_allocation: str = "random",
        mirror_decks: bool = False,
        max_steps: int = 2000,
    ) -> None:
        if not deck_pool:
            raise ValueError("deck_pool must not be empty")
        from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
        from yugioh_rl.deck_selector import DeckSelector

        agent_player_map = {"first": 0, "second": 1, "random": "random"}
        self._agent_player_setting = agent_player_map.get(agent_player, agent_player)

        env_config: dict[str, Any] = {
            "opponent": opponent,
            "agent_player": self._agent_player_setting,
            "collapse_forced": True,
            "max_steps": max_steps,
        }
        if opponent_device is not None:
            env_config["opponent_device"] = opponent_device

        self._env = YuGiOhEnvironment(config=env_config)
        self._deck_pool = deck_pool
        self._selector = DeckSelector(
            pool_size=len(deck_pool), seed=seed, allocation=deck_allocation, mirror=mirror_decks
        )
        self._seed = seed
        self._player_rng = stdlib_random.Random()
        self._last_agent_deck_idx = -1
        self._last_agent_player = -1

    def reset(self, *, episode_idx: int) -> dict[str, np.ndarray]:
        episode_seed = self._seed + episode_idx
        if self._agent_player_setting == "random":
            self._player_rng.seed(episode_seed)
            resolved_player = self._player_rng.randint(0, 1)
        else:
            resolved_player = int(self._agent_player_setting)

        agent_deck_idx, opp_deck_idx = self._selector.select(episode_idx)
        agent_deck = self._deck_pool[agent_deck_idx]
        opp_deck = self._deck_pool[opp_deck_idx]

        if resolved_player == 0:
            deck0, deck1 = agent_deck, opp_deck
        else:
            deck0, deck1 = opp_deck, agent_deck

        self._last_agent_deck_idx = agent_deck_idx
        self._last_agent_player = resolved_player
        obs = self._env.reset(
            seed=episode_seed, deck0=deck0, deck1=deck1, agent_player=resolved_player
        )
        return obs.as_arrays()

    def step(self, action_index: int) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        obs = self._env.step(YuGiOhAction(action_index=action_index))
        np_obs = obs.as_arrays()
        info: dict[str, Any] = {}
        if obs.done:
            info["terminal_reward"] = obs.reward
            info["steps"] = self._env._step_count
            info["agent_deck_idx"] = self._last_agent_deck_idx
            info["turn_count"] = self._env._duel.game_state.turn_count
            info["agent_player"] = self._last_agent_player
            info["timeout"] = self._env._timed_out
            info["opponent_steps"] = self._env._opp_step_count
        return np_obs, obs.reward, obs.done, info

    @property
    def current_msg(self) -> dict | None:
        return self._env.current_msg

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Vectorized environment using multiprocessing
# ---------------------------------------------------------------------------


def limit_worker_blas_threads() -> None:
    """Cap BLAS thread pools in subsequently-spawned worker processes.

    Safe to call in the parent — its own BLAS is already initialised and
    ignores later env-var changes.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker(remote: mp.connection.Connection, env_kwargs: dict) -> None:
    """Worker process: runs a TrainingEnv and responds to commands."""
    env = TrainingEnv(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
            elif cmd == "close":
                env.close()
                remote.send(None)
                break
    except EOFError:
        env.close()


class SubprocVecEnv:
    """Vectorized environment using subprocess workers.

    Each worker runs its own TrainingEnv in a separate process.  The trainer
    holds the policy network and runs batched inference centrally.

    ``collect_rollouts`` encapsulates the per-step inference loop so the
    trainer sees the same interface as the actor-learner vec envs.
    """

    def __init__(
        self,
        num_envs: int,
        deck_pool: list[DeckDict],
        opponent: str = "greedy",
        reward_shaping: bool = True,
        shaping_lp_weight: float = 0.01,
        shaping_card_weight: float = 0.005,
        seed: int = 42,
        agent_player: str = "random",
        deck_allocation: str = "random",
        mirror_decks: bool = False,
        opponent_device: str | None = None,
        master_model: nn.Module | None = None,
        rollout_steps: int = 256,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_sampling: Sampling = "uniform",
        opponent_pool_config: TrainingConfig | None = None,
        max_steps: int = 2000,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self._model = master_model
        self._device = None  # resolved lazily in collect_rollouts
        self._obs: dict[str, np.ndarray] | None = None
        self._closed = False

        ctx = mp.get_context("spawn")
        limit_worker_blas_threads()
        self._remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            env_kwargs = {
                "deck_pool": deck_pool,
                "opponent": opponent,
                "reward_shaping": reward_shaping,
                "shaping_lp_weight": shaping_lp_weight,
                "shaping_card_weight": shaping_card_weight,
                "seed": seed + i * 10000,
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
            p = ctx.Process(target=_worker, args=(child_conn, env_kwargs), daemon=True)
            p.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._workers.append(p)

    def _reset(self) -> dict[str, np.ndarray]:
        """Reset all environments and return stacked observations."""
        for remote in self._remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self._remotes]
        return self._stack_obs(results)

    def _step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with the given actions."""
        for remote, action in zip(self._remotes, actions, strict=True):
            remote.send(("step", int(action)))

        results = [remote.recv() for remote in self._remotes]
        obs_list, rewards, dones, infos = zip(*results, strict=True)

        stacked_obs = self._stack_obs(obs_list)
        return (
            stacked_obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def _reset_done(
        self,
        dones: np.ndarray,
        obs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Substitute new-episode obs at indices where ``dones[i]`` is True.

        Restores wire-level auto-reset semantics that ``TrainingEnv.step()``
        no longer provides.  When ``dones`` is all-False, returns the input
        ``obs`` object unchanged (zero pipe traffic, no copy — callers must
        not mutate the result in that case without copying themselves).
        """
        if not dones.any():
            return obs

        # Sends first, recvs after, so the slowest worker doesn't serialize the rest.
        done_indices = np.flatnonzero(dones)
        for i in done_indices:
            self._remotes[i].send(("reset", None))

        new_obs = {k: v.copy() for k, v in obs.items()}
        for i in done_indices:
            single = self._remotes[i].recv()
            for k in new_obs:
                new_obs[k][i] = single[k]
        return new_obs

    def _stack_obs(self, obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Stack a list of observation dicts into batched arrays."""
        return {key: np.stack([obs[key] for obs in obs_list]) for key in obs_list[0]}

    def collect_rollouts(self) -> list[dict]:
        """Run T steps of batched inference + env.step, return per-env rollouts.

        The trainer's network (``master_model`` from ``__init__``) runs
        batched forward passes on the current device.  Returns the same
        per-env rollout dict format as the actor-learner vec envs so the
        trainer can call ``buffer.ingest_rollouts`` uniformly.
        """
        import torch
        from torch.distributions import Categorical

        from yugioh_rl.network import YuGiOhNet

        model = self._model
        if self._device is None:
            self._device = next(model.parameters()).device
        device = self._device

        if self._obs is None:
            self._obs = self._reset()
        obs = self._obs

        T = self.rollout_steps
        N = self.num_envs

        # Reset hx every rollout: post-update weights make stale hx
        # inconsistent with the current policy.
        hx: HxState = model.init_hx(N, device)

        # Per-env accumulators
        all_obs_cards = np.zeros((T, N, MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        all_obs_global = np.zeros((T, N, GLOBAL_FEATURES), dtype=np.uint8)
        all_obs_actions = np.zeros((T, N, MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
        all_action_mask = np.zeros((T, N, MAX_ACTIONS), dtype=np.int8)
        all_obs_chain = np.zeros((T, N, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES), dtype=np.uint8)
        all_obs_event = np.zeros((T, N, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
        all_actions = np.zeros((T, N), dtype=np.int64)
        all_log_probs = np.zeros((T, N), dtype=np.float32)
        all_values = np.zeros((T, N), dtype=np.float32)
        all_rewards = np.zeros((T, N), dtype=np.float32)
        all_dones = np.zeros((T, N), dtype=np.float32)
        all_infos: list[list[dict]] = [[] for _ in range(N)]

        for t in range(T):
            with torch.no_grad():
                inputs = build_forward_inputs(obs, device=device)
                logits, values, hx_new = model(**inputs, hx=hx)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            next_obs, rewards, dones, infos = self._step(actions_np)
            dones_f = dones.astype(np.float32)

            all_obs_cards[t] = obs["cards"]
            all_obs_global[t] = obs["global_state"]
            all_obs_actions[t] = obs["actions"]
            all_action_mask[t] = obs["action_mask"]
            all_obs_chain[t] = obs["pending_chain"]
            all_obs_event[t] = obs["event_history"]
            all_actions[t] = actions_np
            all_log_probs[t] = log_probs_np
            all_values[t] = values_np
            all_rewards[t] = rewards
            all_dones[t] = dones_f
            for i, info in enumerate(infos):
                all_infos[i].append(info)

            next_obs = self._reset_done(dones, next_obs)

            dones_t = torch.from_numpy(dones_f).to(device)
            hx = YuGiOhNet.mask_hx(hx_new, dones_t)

            obs = next_obs

        self._obs = obs

        # Build per-env rollout dicts matching _pack_rollout's format
        env_indices = [torch.tensor([i], device=device) for i in range(N)]
        rollouts = []
        for i in range(N):
            rollouts.append(
                {
                    "obs_cards": all_obs_cards[:, i],
                    "obs_global": all_obs_global[:, i],
                    "obs_actions": all_obs_actions[:, i],
                    "action_mask": all_action_mask[:, i],
                    "obs_chain": all_obs_chain[:, i],
                    "obs_event": all_obs_event[:, i],
                    "actions": all_actions[:, i],
                    "log_probs": all_log_probs[:, i],
                    "values": all_values[:, i],
                    "rewards": all_rewards[:, i],
                    "dones": all_dones[:, i].astype(bool),
                    "policy_version": 0,
                    "final_obs_cards": obs["cards"][i],
                    "final_obs_global": obs["global_state"][i],
                    "final_obs_actions": obs["actions"][i],
                    "final_action_mask": obs["action_mask"][i],
                    "final_obs_chain": obs["pending_chain"][i],
                    "final_obs_event": obs["event_history"][i],
                    "infos": all_infos[i],
                    "final_hx": YuGiOhNet.slice_hx(hx, env_indices[i]),
                }
            )
        return rollouts

    def publish_weights(self, model) -> int:
        """No-op for SubprocVecEnv — the network is already in-process."""
        self._model = model
        return 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
