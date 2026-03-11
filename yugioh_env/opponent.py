"""Opponent policies for automatic play as Player 1."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from yugioh_env.action_space import ActionMapper
from yugioh_env.constants import (
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_IDLECMD,
    POS_FACEUP_ATTACK,
)

if TYPE_CHECKING:
    pass


class Opponent(ABC):
    """Base class for opponent policies."""

    @abstractmethod
    def select_action(self, msg: dict, mapper: ActionMapper) -> int:
        """Select an action index given the current message and mapper."""
        ...

    @property
    def needs_observation(self) -> bool:
        """Whether this opponent requires full observation arrays to select actions."""
        return False

    def set_observation(self, obs: dict[str, np.ndarray]) -> None:
        """Provide the current observation arrays before calling select_action.

        Only called when needs_observation returns True.
        """

    def reseed(self, seed: int) -> None:
        """Re-seed the opponent's RNG. Override in stochastic subclasses."""


class RandomOpponent(Opponent):
    """Select uniformly random legal actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_action(self, msg: dict, mapper: ActionMapper) -> int:
        n = mapper.num_actions
        if n == 0:
            return 0
        return self._rng.randint(0, n - 1)


class GreedyOpponent(Opponent):
    """Simple heuristic opponent.

    Strategy:
    - In idle cmd: summon strongest monster, set spells/traps, then enter battle
    - In battle cmd: attack with strongest, then end
    - For other messages: pick first valid option
    """

    def select_action(self, msg: dict, mapper: ActionMapper) -> int:
        n = mapper.num_actions
        if n == 0:
            return 0
        if n == 1:
            return 0

        msg_type = msg.get("msg_type")

        if msg_type == MSG_SELECT_IDLECMD:
            return self._greedy_idle(msg, mapper)
        elif msg_type == MSG_SELECT_BATTLECMD:
            return self._greedy_battle(msg, mapper)
        else:
            return 0

    def _greedy_idle(self, msg: dict, mapper: ActionMapper) -> int:
        """Greedy idle: summon best monster > set S/T > activate > go to BP > end."""
        # Try to summon the monster with highest ATK
        summonable = msg.get("summonable", [])
        if summonable:
            best_idx = 0
            # First summonable action in the mapper
            return 0

        # Try special summon
        if msg.get("sp_summonable"):
            sp_start = len(msg.get("summonable", []))
            return sp_start

        # Try setting spells/traps
        sset = msg.get("sset", [])
        if sset:
            offset = (
                len(msg.get("summonable", []))
                + len(msg.get("sp_summonable", []))
                + len(msg.get("repositionable", []))
                + len(msg.get("mset", []))
            )
            return min(offset, mapper.num_actions - 1)

        # Try to enter battle phase
        activatable_count = (
            len(msg.get("summonable", []))
            + len(msg.get("sp_summonable", []))
            + len(msg.get("repositionable", []))
            + len(msg.get("mset", []))
            + len(msg.get("sset", []))
            + len(msg.get("activatable", []))
        )
        if msg.get("to_bp"):
            return min(activatable_count, mapper.num_actions - 1)

        # End phase
        return mapper.num_actions - 1

    def _greedy_battle(self, msg: dict, mapper: ActionMapper) -> int:
        """Greedy battle: attack if possible, then end."""
        attackable = msg.get("attackable", [])
        act_count = len(msg.get("activatable", []))
        if attackable:
            return act_count  # First attack action
        # Go to M2 or EP
        return mapper.num_actions - 1


class ModelOpponent(Opponent):
    """Opponent that uses a trained YuGiOhNet checkpoint to select actions.

    Requires torch and yugioh_rl to be installed (``pip install -e ".[train]"``).
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        import torch
        from yugioh_rl.config import TrainingConfig
        from yugioh_rl.network import YuGiOhNet

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config: TrainingConfig = checkpoint["config"]
        self._network = YuGiOhNet(config)
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._network.to(device)
        self._network.eval()
        self._device = torch.device(device)
        self._obs: dict[str, np.ndarray] | None = None

    @property
    def needs_observation(self) -> bool:
        return True

    def set_observation(self, obs: dict[str, np.ndarray]) -> None:
        self._obs = obs

    def select_action(self, msg: dict, mapper: ActionMapper) -> int:
        import torch

        if self._obs is None:
            return 0

        obs = self._obs
        t_cards = torch.from_numpy(obs["cards"]).unsqueeze(0).to(self._device)
        t_global = torch.from_numpy(obs["global_state"]).unsqueeze(0).to(self._device)
        t_actions = torch.from_numpy(obs["actions"]).unsqueeze(0).to(self._device)
        t_mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, _ = self._network(t_cards, t_global, t_actions, t_mask)
            action = logits.argmax(dim=-1).item()

        return min(action, mapper.num_actions - 1)

    def reseed(self, seed: int) -> None:
        pass  # Deterministic policy
