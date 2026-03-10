"""Training environment wrappers: single-env and vectorized."""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from typing import Any

from yugioh_env.models import YuGiOhAction


def _obs_to_numpy(obs) -> dict[str, np.ndarray]:
    """Convert a YuGiOhObservation (with Python lists) back to numpy arrays."""
    return {
        "cards": np.array(obs.cards, dtype=np.uint8).reshape(200, 42) if obs.cards else np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.array(obs.global_state, dtype=np.uint8).reshape(20) if obs.global_state else np.zeros(20, dtype=np.uint8),
        "actions": np.array(obs.actions, dtype=np.uint8).reshape(32, 12) if obs.actions else np.zeros((32, 12), dtype=np.uint8),
        "action_mask": np.array(obs.action_mask, dtype=np.int8).reshape(32) if obs.action_mask else np.zeros(32, dtype=np.int8),
    }


class TrainingEnv:
    """Single-env wrapper around YuGiOhEnvironment for training.

    Bypasses HTTP by calling the environment directly in-process.
    Provides numpy observations, reward shaping, and auto-reset.
    """

    def __init__(
        self,
        deck_path: str = "assets/decks/starter.ydk",
        opponent_type: str = "greedy",
        reward_shaping: bool = True,
        shaping_lp_weight: float = 0.01,
        shaping_card_weight: float = 0.005,
        seed: int = 42,
    ) -> None:
        from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

        self._env = YuGiOhEnvironment(config={
            "deck_path": deck_path,
            "opponent_type": opponent_type,
        })
        self._reward_shaping = reward_shaping
        self._lp_weight = shaping_lp_weight
        self._card_weight = shaping_card_weight
        self._seed = seed
        self._episode_count = 0

        # State for reward shaping
        self._prev_my_lp = 0
        self._prev_opp_lp = 0
        self._prev_advantage = 0

    def reset(self) -> dict[str, np.ndarray]:
        """Reset the environment and return initial observation."""
        self._episode_count += 1
        obs = self._env.reset(seed=self._seed + self._episode_count)
        np_obs = _obs_to_numpy(obs)

        # Initialize shaping state
        gs = np_obs["global_state"]
        self._prev_my_lp = int(gs[0]) + int(gs[1]) * 256
        self._prev_opp_lp = int(gs[2]) + int(gs[3]) * 256
        self._prev_advantage = self._compute_advantage(gs)

        return np_obs

    def step(self, action_index: int) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Step the environment with an action.

        Returns:
            obs: numpy observation dict
            reward: shaped or terminal reward
            done: whether episode ended
            info: extra info (terminal_reward, episode_length on done)
        """
        obs = self._env.step(YuGiOhAction(action_index=action_index))
        np_obs = _obs_to_numpy(obs)
        info: dict[str, Any] = {}

        reward = obs.reward
        done = obs.done

        if done:
            info["terminal_reward"] = reward
            info["episode_length"] = self._env._step_count

            # Auto-reset: return first obs of new episode
            np_obs = self.reset()
        elif self._reward_shaping:
            reward += self._compute_shaping(np_obs["global_state"])

        return np_obs, reward, done, info

    def _compute_advantage(self, gs: np.ndarray) -> int:
        """Compute card advantage from global state zone counts."""
        # bytes 10=my_hand, 9=my_deck is not useful; use hand + field
        my_hand = int(gs[10])
        my_field = int(gs[11]) + int(gs[12])  # Actually: 9=deck,10=hand,11=grave...
        # Correct mapping from observation.py global state layout:
        # 9=my_deck, 10=my_hand, 11=my_grave, 12=my_banished, 13=my_extra
        # 14=opp_deck, 15=opp_hand, 16=opp_grave, 17=opp_banished, 18=opp_extra
        # We approximate "field" cards from LP - deck - hand - grave - banished - extra
        # Actually, zone counts in global state don't include mzone/szone directly.
        # But we do have hand counts. Let's use hand differential as a simpler proxy.
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
        deck_path: str = "assets/decks/starter.ydk",
        opponent_type: str = "greedy",
        reward_shaping: bool = True,
        shaping_lp_weight: float = 0.01,
        shaping_card_weight: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.num_envs = num_envs
        self._closed = False

        ctx = mp.get_context("spawn")
        self._remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            env_kwargs = {
                "deck_path": deck_path,
                "opponent_type": opponent_type,
                "reward_shaping": reward_shaping,
                "shaping_lp_weight": shaping_lp_weight,
                "shaping_card_weight": shaping_card_weight,
                "seed": seed + i * 10000,
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
