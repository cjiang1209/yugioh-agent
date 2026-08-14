"""Opponent policies for automatic play as Player 1.

Also exposes:
- ``parse_opponent_spec(spec)`` — split "greedy" / "random" / "model:path" strings.
- ``make_opponent(spec, seed, device)`` — factory used by both the HTTP server
  (``YuGiOhEnvironment.__init__``) and the eval module.

Every opponent consumes the canonical ``YuGiOhObservation`` — the same shape
the agent sees. ``Opponent.needs_board_state`` gates how much of it gets
built, not whether it is passed.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from yugioh_core.action_categories import IDLE_SP_SUMMON, IDLE_SSET, IDLE_SUMMON
from yugioh_core.constants import (
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_IDLECMD,
)
from yugioh_env.models import Attack, CardCommand, PhaseChange, YuGiOhObservation


@dataclass(frozen=True)
class Inference:
    """Readouts from the forward pass that chose an action.

    Travels back with that action, so a caller holding one holds the other.
    """

    value: float
    """Raw value-head output for the acting seat's position."""

    action_probs: list[float]
    """Policy probabilities over the legal actions, in engine index order."""


class Opponent(ABC):
    """Base class for opponent policies."""

    @property
    @abstractmethod
    def needs_board_state(self) -> bool:
        """Does ``select_action`` read the BOARD half of the observation
        (``cards`` / ``global_state`` / ``pending_chain`` / ``event_history``)?

        The environment builds the opponent-seat observation accordingly,
        skipping the engine ``query_location`` calls that dominate its cost.
        A ``YuGiOhObservation`` arrives either way; the skipped fields hold
        shaped zeros rather than the real board.
        """

    @abstractmethod
    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        """Choose an action for the current observation.

        Returns the chosen index and, for a policy with a value head, the
        readouts from the same forward pass. Implementations without one
        return ``None`` as the second element.
        """
        ...

    def reseed(self, seed: int) -> None:  # noqa: B027
        """Re-seed the opponent's RNG. Override in stochastic subclasses."""


class RandomOpponent(Opponent):
    """Select uniformly random legal actions."""

    @property
    def needs_board_state(self) -> bool:
        return False  # reads only the action count

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        num_actions = obs.num_actions
        if num_actions == 0:
            return 0, None
        return self._rng.randint(0, num_actions - 1), None


class GreedyOpponent(Opponent):
    """Simple heuristic opponent.

    Idle: summon > special summon > set S/T > battle phase > last slot.
    Battle: attack > last slot. Any other prompt: slot 0.

    Takes the FIRST descriptor matching each category, not the strongest --
    ranking would need the board, which ``needs_board_state = False`` skips.
    Scanning descriptors also means only actions the loop filter left are
    considered.
    """

    @property
    def needs_board_state(self) -> bool:
        return False  # reads descriptors and prompt_meta

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        num_actions = obs.num_actions
        if num_actions <= 1:
            return 0, None

        msg_type = obs.msg_type
        descriptors = obs.action_descriptors

        if msg_type == MSG_SELECT_IDLECMD:
            return self._greedy_idle(descriptors), None
        elif msg_type == MSG_SELECT_BATTLECMD:
            return self._greedy_battle(descriptors), None
        else:
            return 0, None

    @staticmethod
    def _first_command(descriptors: list, command: int) -> int | None:
        for i, d in enumerate(descriptors):
            if isinstance(d, CardCommand) and d.command == command:
                return i
        return None

    @staticmethod
    def _first_phase(descriptors: list, to: str) -> int | None:
        for i, d in enumerate(descriptors):
            if isinstance(d, PhaseChange) and d.to == to:
                return i
        return None

    @staticmethod
    def _first_attack(descriptors: list) -> int | None:
        for i, d in enumerate(descriptors):
            if isinstance(d, Attack):
                return i
        return None

    def _greedy_idle(self, descriptors: list) -> int:
        for command in (IDLE_SUMMON, IDLE_SP_SUMMON, IDLE_SSET):
            idx = self._first_command(descriptors, command)
            if idx is not None:
                return idx

        idx = self._first_phase(descriptors, "bp")
        if idx is not None:
            return idx

        return len(descriptors) - 1

    def _greedy_battle(self, descriptors: list) -> int:
        idx = self._first_attack(descriptors)
        if idx is not None:
            return idx
        return len(descriptors) - 1


class NetworkOpponent(Opponent):
    """Opponent that uses an already-loaded ``YuGiOhNet`` for greedy argmax inference.

    Used by in-training eval (the trainer passes ``self.network`` directly,
    avoiding a per-cycle checkpoint load) and by ``ModelOpponent`` (which loads
    a checkpoint from disk and delegates here).

    Requires torch and yugioh_rl to be installed (``pip install -e ".[train]"``).
    """

    @property
    def needs_board_state(self) -> bool:
        return True  # the network reads every board field

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
        # Per-episode recurrent state.  reseed() — called per duel by both
        # the HTTP env and the eval loop — re-zeros it.
        self._hx = self._network.init_hx(1, self._device)

    @property
    def network(self):
        """The wrapped policy network."""
        return self._network

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        import torch

        from yugioh_rl.obs_encoder import encode_observation
        from yugioh_rl.policy_inputs import build_forward_inputs

        inputs = build_forward_inputs(
            encode_observation(obs), device=self._device, add_batch_dim=True
        )

        with torch.no_grad():
            logits, values, self._hx = self._network(**inputs, hx=self._hx)
            masked = logits.masked_fill(~inputs["action_mask"].bool(), float("-inf"))
            # One softmax per branch, reported as-is: `action_probs` is the
            # distribution the action came from, so a temperature that shapes
            # the sampling shapes the report too. Each branch enqueues it before
            # its `.item()`, so the sync that reads the action also waits out
            # this kernel -- the reads below stall on no GPU work, though each
            # is still its own device-to-host copy.
            if self._stochastic:
                probs = torch.softmax(masked / self._temperature, dim=-1)
                action = int(torch.multinomial(probs[0], 1).item())
            else:
                probs = torch.softmax(masked, dim=-1)
                action = int(masked.argmax(dim=-1).item())
            inference = Inference(
                value=float(values[0].item()),
                # A contiguous prefix slice is only correct because the mask
                # is a dense prefix: it is built from the descriptor count.
                action_probs=probs[0, : obs.num_actions].tolist(),
            )

        return action, inference

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
    def needs_board_state(self) -> bool:
        return self._impl.needs_board_state

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        return self._impl.select_action(obs)

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
    - ``"ygo-agent"`` → ``("ygo-agent", "")``
    - ``"ygo-agent:http://host:3000"`` → ``("ygo-agent", "http://host:3000")``

    Does not validate that checkpoint files exist — that's a CLI-layer concern.
    """
    if spec.startswith("model:"):
        return "model", spec[len("model:") :]
    if spec.startswith("ygo-agent:"):
        return "ygo-agent", spec[len("ygo-agent:") :]
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
    - ``"ygo-agent"`` → ``YGOAgentOpponent()`` (default localhost:3000)
    - ``"ygo-agent:url"`` → ``YGOAgentOpponent(url)``

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
    if opponent_type == "ygo-agent":
        from yugioh_env.ygo_agent import YGOAgentOpponent

        return YGOAgentOpponent(checkpoint) if checkpoint else YGOAgentOpponent()
    raise ValueError(f"unknown opponent: {spec!r}")
