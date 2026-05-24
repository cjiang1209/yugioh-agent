"""Duel agents for the MUD bot.

An ``Agent`` chooses an action given a parsed prompt.  The
``ActionTranslator`` (see ``action_translator.py``) converts the int
return value into the MUD text command.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Protocol

from yugioh_mud.text_parser import ParsedPrompt, PromptType

if TYPE_CHECKING:
    from yugioh_mud.game_state import MUDGameState

# ---------------------------------------------------------------------------
# Special action constants (negative = meta-commands)
# ---------------------------------------------------------------------------

END_PHASE = -1
CANCEL = -2
DECLINE = -3
BACK = -4
FINISH = -5


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


class Agent(Protocol):
    """Decides *what* to do given a parsed prompt.

    Returns an int:
    * ``>= 0`` — index into ``prompt.options`` (0-based)
    * negative — one of the special constants above
    """

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int: ...


# ---------------------------------------------------------------------------
# PassiveAgent — always end phase, decline effects, cancel chains
# ---------------------------------------------------------------------------


class PassiveAgent:
    """Passive bot: ends phases, declines effects, cancels chains.

    Duels end by deck-out (neither player takes voluntary actions).
    For mandatory selections (e.g. forced position, forced tribute)
    it picks the first available option.
    """

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int:
        pt = prompt.prompt_type

        if pt in (PromptType.IDLE_CMD, PromptType.BATTLE_MENU):
            return END_PHASE

        if pt == PromptType.SELECT_CHAIN:
            return CANCEL if prompt.cancelable else 0
        if pt == PromptType.SELECT_EFFECTYN:
            return DECLINE
        if pt == PromptType.SELECT_YESNO:
            return DECLINE

        if pt in (PromptType.SELECT_CARD, PromptType.SELECT_TRIBUTE, PromptType.SELECT_SUM):
            return 0  # translator picks first min_select options

        if pt == PromptType.SELECT_POSITION:
            return 0
        if pt == PromptType.SELECT_PLACE:
            return 0
        if pt == PromptType.SELECT_OPTION:
            return 0

        if pt == PromptType.SELECT_COUNTER:
            return 0  # translator sends all zeros

        if pt == PromptType.SELECT_UNSELECT:
            return FINISH if prompt.finishable else 0

        if pt in (PromptType.ANNOUNCE_RACE, PromptType.ANNOUNCE_ATTRIB):
            return 0
        if pt == PromptType.ANNOUNCE_NUMBER:
            return 0
        if pt == PromptType.ANNOUNCE_CARD:
            return CANCEL
        if pt == PromptType.SORT_CARD:
            return CANCEL

        # UNKNOWN or unhandled
        return CANCEL


# ---------------------------------------------------------------------------
# RandomAgent — picks uniformly from legal actions
# ---------------------------------------------------------------------------


class RandomAgent:
    """Random bot: picks uniformly from legal actions.

    Uses an instance-level ``random.Random`` for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._log = logging.getLogger(__name__)

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int:
        pt = prompt.prompt_type
        n = len(prompt.options)

        # Idle/battle: use structured_actions (populated by handlers)
        if pt in (PromptType.IDLE_CMD, PromptType.BATTLE_MENU):
            sa = prompt.structured_actions
            if not sa:
                return END_PHASE
            choices = list(range(len(sa))) + [END_PHASE]
            self._log.debug("[RandomAgent] %s structured_actions=%d", pt.name, len(sa))
            return self._rng.choice(choices)

        # Chain with optional cancel
        if pt == PromptType.SELECT_CHAIN:
            if n == 0:
                return CANCEL
            if prompt.cancelable:
                choices = list(range(n)) + [CANCEL]
                return self._rng.choice(choices)
            return self._rng.randrange(n)

        # Effect Y/N and Yes/No
        if pt in (PromptType.SELECT_EFFECTYN, PromptType.SELECT_YESNO):
            return self._rng.choice([0, DECLINE])

        # Unselect with finish
        if pt == PromptType.SELECT_UNSELECT:
            if n == 0:
                return FINISH
            if prompt.finishable:
                choices = list(range(n)) + [FINISH]
                return self._rng.choice(choices)
            return self._rng.randrange(n)

        # Sort/announce card — always cancel
        if pt in (PromptType.SORT_CARD, PromptType.ANNOUNCE_CARD, PromptType.UNKNOWN):
            return CANCEL

        # Standard selections: random index
        if n > 0:
            return self._rng.randrange(n)
        return 0


# ---------------------------------------------------------------------------
# Model action mapping (module-level for testability without torch)
# ---------------------------------------------------------------------------


def map_model_action(action_idx: int, prompt: ParsedPrompt) -> int:
    """Map a model's raw action index to an Agent return value.

    Translates the integer action output from the RL network into the
    appropriate agent constant for the given prompt type.
    """
    pt = prompt.prompt_type
    n = len(prompt.options)

    if pt in (PromptType.IDLE_CMD, PromptType.BATTLE_MENU):
        sa = prompt.structured_actions
        if action_idx >= len(sa):
            return END_PHASE
        # Return the index directly — the cmd_handler dispatches phase
        # transitions (to_bp, to_ep, to_m2) via StructuredAction.sub_action.
        return action_idx

    if pt in (PromptType.SELECT_EFFECTYN, PromptType.SELECT_YESNO):
        return 0 if action_idx == 0 else DECLINE

    if pt == PromptType.SELECT_CHAIN:
        return action_idx if action_idx < n else CANCEL

    if pt == PromptType.SELECT_UNSELECT:
        return action_idx if action_idx < n else FINISH

    # Generic selections: clamp to valid range
    if n > 0:
        return min(action_idx, n - 1)

    return CANCEL


# ---------------------------------------------------------------------------
# ModelAgent — uses a trained YuGiOhNet checkpoint
# ---------------------------------------------------------------------------


class ModelAgent:
    """Agent that uses a trained YuGiOhNet checkpoint to select actions.

    Requires torch and yugioh_rl to be installed (``pip install -e ".[train]"``).
    """

    def __init__(
        self,
        checkpoint_path: str,
        db_path: str,
        device: str = "cpu",
    ) -> None:
        import torch

        from yugioh_core.card_database import CardDatabase
        from yugioh_mud.observation import MUDObservationBuilder
        from yugioh_rl.config import TrainingConfig, normalize_legacy_config
        from yugioh_rl.network import YuGiOhNet

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config: TrainingConfig = normalize_legacy_config(checkpoint["config"])
        self._network = YuGiOhNet.from_state_dict(config, checkpoint["model_state_dict"])
        self._network.to(device)
        self._network.eval()
        self._device = torch.device(device)
        self._hx = self._network.init_hx(1, self._device)

        card_db = CardDatabase(db_path)
        self._obs_builder = MUDObservationBuilder(card_db)
        self._fallback = PassiveAgent()
        self._log = logging.getLogger(__name__)

    def reset_hidden_state(self) -> None:
        """Re-zero recurrent state at the start of a new duel.

        Called by ``MUDProtocol`` on entry to ``State.DUEL``.  No-op for
        feed-forward networks (``init_hx`` returns ``None``).
        """
        self._hx = self._network.init_hx(1, self._device)

    def choose(
        self,
        prompt: ParsedPrompt,
        game_state: MUDGameState | None = None,
    ) -> int:
        if game_state is None:
            return self._fallback.choose(prompt)

        import torch

        obs = self._obs_builder.build(game_state, prompt)
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
            action_idx = logits.argmax(dim=-1).item()

        return map_model_action(action_idx, prompt)
