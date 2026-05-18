"""Training environment wrappers: single-env and vectorized."""

from __future__ import annotations

import multiprocessing as mp
import random as stdlib_random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_env.models import YuGiOhAction

if TYPE_CHECKING:
    from yugioh_rl.config import TrainingConfig

DeckDict = dict[str, list[int]]  # {"main": [int, ...], "extra": [int, ...]}


def parse_deck_pool(deck_paths: list[str]) -> list[DeckDict]:
    """Pre-parse .ydk files into dicts for passing to worker processes.

    Called once in the main process; the resulting list is picklable and
    can be sent to TrainingEnv workers without further file I/O.
    """
    from yugioh_env.deck_parser import parse_ydk
    return [parse_ydk(p) for p in deck_paths]


def _obs_to_numpy(obs) -> dict[str, np.ndarray]:
    """Convert a YuGiOhObservation (with Python lists) back to numpy arrays."""
    return {
        "cards": (
            np.array(obs.cards, dtype=np.uint8).reshape(MAX_CARDS, CARD_FEATURES)
            if obs.cards else np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        ),
        "global_state": (
            np.array(obs.global_state, dtype=np.uint8).reshape(GLOBAL_FEATURES)
            if obs.global_state else np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
        ),
        "actions": (
            np.array(obs.actions, dtype=np.uint8).reshape(MAX_ACTIONS, ACTION_FEATURES)
            if obs.actions else np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
        ),
        "action_mask": (
            np.array(obs.action_mask, dtype=np.int8).reshape(MAX_ACTIONS)
            if obs.action_mask else np.zeros(MAX_ACTIONS, dtype=np.int8)
        ),
    }


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
        opponent_device: str | None = None,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_config: "TrainingConfig | None" = None,
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

        # Deck pool and sampling RNG.  ``_deck_rng`` is reseeded per-episode
        # in reset() so deck draws are a pure function of (seed, episode_count)
        # — required for episode-shard parallelism.
        self._deck_pool = deck_pool
        self._deck_rng = stdlib_random.Random(seed)
        self._player_rng = stdlib_random.Random()
        self._last_agent_deck_idx = -1

        # State for reward shaping
        self._prev_my_lp = 0
        self._prev_opp_lp = 0
        self._prev_advantage = 0

        self._opponent_pool = None
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

        self._deck_rng = stdlib_random.Random(episode_seed)
        agent_deck_idx = self._deck_rng.randrange(len(self._deck_pool))
        opp_deck_idx = self._deck_rng.randrange(len(self._deck_pool))
        agent_deck = self._deck_pool[agent_deck_idx]
        opp_deck = self._deck_pool[opp_deck_idx]

        # Map agent/opponent to engine player 0/1
        if resolved_player == 0:
            deck0, deck1 = agent_deck, opp_deck
        else:
            deck0, deck1 = opp_deck, agent_deck

        self._last_agent_deck_idx = agent_deck_idx

        if self._opponent_pool is not None:
            new_opp = self._opponent_pool.sample()
            self._env.set_opponent(new_opp)

        obs = self._env.reset(
            seed=episode_seed,
            deck0=deck0,
            deck1=deck1,
            agent_player=resolved_player,
        )
        np_obs = _obs_to_numpy(obs)

        # Initialize shaping state
        gs = np_obs["global_state"]
        self._prev_my_lp = int(gs[0]) + int(gs[1]) * 256
        self._prev_opp_lp = int(gs[2]) + int(gs[3]) * 256
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
                  ``episode_length``, and ``agent_deck_idx``
        """
        obs = self._env.step(YuGiOhAction(action_index=action_index))
        np_obs = _obs_to_numpy(obs)
        info: dict[str, Any] = {}

        reward = obs.reward
        done = obs.done

        if done:
            info["terminal_reward"] = reward
            info["episode_length"] = self._env._step_count
            info["agent_deck_idx"] = self._last_agent_deck_idx
            # No auto-reset.  Caller chooses when to begin the next episode
            # (vec-env workers do it eagerly to preserve wire behavior;
            # eval workers reset to a specific episode_idx for the next task).
        elif self._reward_shaping:
            reward += self._compute_shaping(np_obs["global_state"])

        return np_obs, reward, done, info

    def _compute_advantage(self, gs: np.ndarray) -> int:
        """Compute card advantage from global state zone counts."""
        # Correct mapping from observation.py global state layout:
        # 9=my_deck, 10=my_hand, 11=my_grave, 12=my_banished, 13=my_extra
        # 14=opp_deck, 15=opp_hand, 16=opp_grave, 17=opp_banished, 18=opp_extra
        my_hand = int(gs[10])
        opp_hand = int(gs[15])
        return my_hand - opp_hand

    def _compute_shaping(self, gs: np.ndarray) -> float:
        """Compute reward shaping based on LP and card advantage deltas."""
        my_lp = int(gs[0]) + int(gs[1]) * 256
        opp_lp = int(gs[2]) + int(gs[3]) * 256
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


# ---------------------------------------------------------------------------
# Vectorized environment using multiprocessing
# ---------------------------------------------------------------------------

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

    Each worker runs its own TrainingEnv in a separate process.
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
        opponent_device: str | None = None,
        opponent_pool_handles: dict | None = None,
        opponent_pool_temperature: float = 1.0,
        opponent_pool_config: "TrainingConfig | None" = None,
    ) -> None:
        self.num_envs = num_envs
        self._closed = False

        ctx = mp.get_context("spawn")
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
                "opponent_device": opponent_device,
                "opponent_pool_handles": opponent_pool_handles,
                "opponent_pool_temperature": opponent_pool_temperature,
                "opponent_pool_config": opponent_pool_config,
            }
            p = ctx.Process(target=_worker, args=(child_conn, env_kwargs), daemon=True)
            p.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._workers.append(p)

    def reset(self) -> dict[str, np.ndarray]:
        """Reset all environments and return stacked observations."""
        for remote in self._remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self._remotes]
        return self._stack_obs(results)

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with the given actions.

        Returns the **terminal** observation for envs that finished this
        step (``dones[i] == True``).  Callers that want the next-episode
        first observation at those indices must follow up with
        :meth:`reset_done`.  The previous auto-reset behavior was moved out
        of ``TrainingEnv.step`` so eval workers can address specific
        episodes — production training paths restore the old wire shape via
        the explicit :meth:`reset_done` call.

        Args:
            actions: (num_envs,) int array of action indices

        Returns:
            obs: stacked numpy dict — each value has leading dim num_envs
            rewards: (num_envs,) float
            dones: (num_envs,) bool
            infos: list of info dicts
        """
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", int(action)))

        results = [remote.recv() for remote in self._remotes]
        obs_list, rewards, dones, infos = zip(*results)

        stacked_obs = self._stack_obs(obs_list)
        return (
            stacked_obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def reset_done(
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
        return {
            key: np.stack([obs[key] for obs in obs_list])
            for key in obs_list[0]
        }

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
