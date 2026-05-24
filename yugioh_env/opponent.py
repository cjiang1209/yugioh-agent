"""Opponent policies for automatic play as Player 1.

Also exposes:
- ``parse_opponent_spec(spec)`` — split "greedy" / "random" / "model:path" strings.
- ``make_opponent(spec, seed, device)`` — factory used by both the HTTP server
  (``YuGiOhEnvironment.__init__``) and the eval module.

The ``select_action`` ABC takes ``num_actions: int`` rather than the full
``ActionMapper`` because every subclass only ever needs the count to clamp its
chosen index — exposing more of the mapper was over-scoped.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

from yugioh_core.constants import (
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_IDLECMD,
)


class Opponent(ABC):
    """Base class for opponent policies."""

    @abstractmethod
    def select_action(self, msg: dict, num_actions: int) -> int:
        """Select an action index in ``[0, num_actions)`` given the current message."""
        ...

    @property
    def needs_observation(self) -> bool:
        """Whether this opponent requires full observation arrays to select actions."""
        return False

    def set_observation(self, obs: dict[str, np.ndarray]) -> None:  # noqa: B027
        """Provide the current observation arrays before calling select_action.

        Only called when needs_observation returns True.
        """

    def reseed(self, seed: int) -> None:  # noqa: B027
        """Re-seed the opponent's RNG. Override in stochastic subclasses."""


class RandomOpponent(Opponent):
    """Select uniformly random legal actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_action(self, msg: dict, num_actions: int) -> int:
        if num_actions == 0:
            return 0
        return self._rng.randint(0, num_actions - 1)


class GreedyOpponent(Opponent):
    """Simple heuristic opponent.

    Strategy:
    - In idle cmd: summon strongest monster, set spells/traps, then enter battle
    - In battle cmd: attack with strongest, then end
    - For other messages: pick first valid option
    """

    def select_action(self, msg: dict, num_actions: int) -> int:
        if num_actions == 0:
            return 0
        if num_actions == 1:
            return 0

        msg_type = msg.get("msg_type")

        if msg_type == MSG_SELECT_IDLECMD:
            return self._greedy_idle(msg, num_actions)
        elif msg_type == MSG_SELECT_BATTLECMD:
            return self._greedy_battle(msg, num_actions)
        else:
            return 0

    def _greedy_idle(self, msg: dict, num_actions: int) -> int:
        """Greedy idle: summon > set S/T > go to BP > end."""
        if msg.get("summonable"):
            return 0

        if msg.get("sp_summonable"):
            return len(msg.get("summonable", []))

        if msg.get("sset"):
            offset = (
                len(msg.get("summonable", []))
                + len(msg.get("sp_summonable", []))
                + len(msg.get("repositionable", []))
                + len(msg.get("mset", []))
            )
            return min(offset, num_actions - 1)

        activatable_count = (
            len(msg.get("summonable", []))
            + len(msg.get("sp_summonable", []))
            + len(msg.get("repositionable", []))
            + len(msg.get("mset", []))
            + len(msg.get("sset", []))
            + len(msg.get("activatable", []))
        )
        if msg.get("to_bp"):
            return min(activatable_count, num_actions - 1)

        return num_actions - 1

    def _greedy_battle(self, msg: dict, num_actions: int) -> int:
        """Greedy battle: attack if possible, then end."""
        act_count = len(msg.get("activatable", []))
        if msg.get("attackable"):
            return act_count
        return num_actions - 1


class NetworkOpponent(Opponent):
    """Opponent that uses an already-loaded ``YuGiOhNet`` for greedy argmax inference.

    Used by in-training eval (the trainer passes ``self.network`` directly,
    avoiding a per-cycle checkpoint load) and by ``ModelOpponent`` (which loads
    a checkpoint from disk and delegates here).

    Requires torch and yugioh_rl to be installed (``pip install -e ".[train]"``).
    """

    def __init__(
        self,
        network,
        device: str = "cpu",
        *,
        stochastic: bool = False,
        temperature: float = 1.0,
    ) -> None:
        import torch

        self._network = network
        self._device = torch.device(device)
        self._stochastic = stochastic
        self._temperature = temperature
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        self._obs: dict[str, np.ndarray] | None = None
        # Per-episode recurrent state.  reseed() — called per duel by both
        # the HTTP env and the eval loop — re-zeros it.
        self._hx = self._network.init_hx(1, self._device)

    @property
    def needs_observation(self) -> bool:
        return True

    def set_observation(self, obs: dict[str, np.ndarray]) -> None:
        self._obs = obs

    @property
    def network(self):
        """The wrapped policy network."""
        return self._network

    def select_action(self, msg: dict, num_actions: int) -> int:
        import torch

        if self._obs is None:
            return 0

        obs = self._obs
        t_cards = torch.from_numpy(obs["cards"]).unsqueeze(0).to(self._device)
        t_global = torch.from_numpy(obs["global_state"]).unsqueeze(0).to(self._device)
        t_actions = torch.from_numpy(obs["actions"]).unsqueeze(0).to(self._device)
        t_mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, _, self._hx = self._network(
                t_cards,
                t_global,
                t_actions,
                t_mask,
                hx=self._hx,
            )
            masked = logits.masked_fill(~t_mask.bool(), float("-inf"))
            if self._stochastic:
                probs = torch.softmax(masked / self._temperature, dim=-1)
                action = int(torch.multinomial(probs[0], 1).item())
            else:
                action = int(masked.argmax(dim=-1).item())

        return min(action, num_actions - 1)

    def reseed(self, seed: int) -> None:
        # Resets per-duel recurrent state. Stochastic-mode sampling uses
        # the global torch RNG and is not reseeded here.
        self._hx = self._network.init_hx(1, self._device)


class ModelOpponent(Opponent):
    """Opponent that loads a trained ``YuGiOhNet`` checkpoint and delegates to ``NetworkOpponent``.

    Requires torch and yugioh_rl to be installed (``pip install -e ".[train]"``).
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        import torch

        from yugioh_rl.config import normalize_legacy_config
        from yugioh_rl.network import YuGiOhNet

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = normalize_legacy_config(checkpoint["config"])
        network = YuGiOhNet.from_state_dict(config, checkpoint["model_state_dict"])
        network.to(device)
        network.eval()
        self._impl = NetworkOpponent(network, device=device)

    @property
    def needs_observation(self) -> bool:
        return True

    def set_observation(self, obs: dict[str, np.ndarray]) -> None:
        self._impl.set_observation(obs)

    def select_action(self, msg: dict, num_actions: int) -> int:
        return self._impl.select_action(msg, num_actions)

    def reseed(self, seed: int) -> None:
        self._impl.reseed(seed)


# ---------------------------------------------------------------------------
# Opponent-spec parsing and factory
# ---------------------------------------------------------------------------


def parse_opponent_spec(spec: str) -> tuple[str, str]:
    """Parse an opponent spec string.

    Returns ``(opponent_type, checkpoint_path)``:
    - ``"greedy"`` → ``("greedy", "")``
    - ``"random"`` → ``("random", "")``
    - ``"model:/p.pt"`` → ``("model", "/p.pt")``

    Does not validate that checkpoint files exist — that's a CLI-layer concern.
    """
    if spec.startswith("model:"):
        return "model", spec[len("model:") :]
    return spec, ""


def make_opponent(
    spec: str,
    *,
    seed: int | None = None,
    device: str = "cpu",
) -> Opponent:
    """Construct an ``Opponent`` from a spec string.

    - ``"greedy"`` → ``GreedyOpponent()``
    - ``"random"`` → ``RandomOpponent(seed=seed)``
    - ``"model:path.pt"`` → ``ModelOpponent(path, device=device)``

    Raises ``ValueError`` on empty model path or unknown kind. Checkpoint
    file-existence is not validated here — callers that want a clean CLI error
    message should validate the path before calling this factory.
    """
    opponent_type, checkpoint = parse_opponent_spec(spec)
    if opponent_type == "model":
        if not checkpoint:
            raise ValueError(
                "model opponent requires a checkpoint path (e.g. 'model:path/to/ckpt.pt')"
            )
        return ModelOpponent(checkpoint, device=device)
    if opponent_type == "greedy":
        return GreedyOpponent()
    if opponent_type == "random":
        return RandomOpponent(seed=seed)
    raise ValueError(f"unknown opponent: {spec!r}")
