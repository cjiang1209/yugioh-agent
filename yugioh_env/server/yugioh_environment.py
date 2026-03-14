"""OpenEnv Environment subclass for Yu-Gi-Oh! (server-side)."""

from __future__ import annotations

import logging
import os
import random as stdlib_random
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation

from yugioh_env.action_space import ActionMapper
from yugioh_env.card_database import CardDatabase
from yugioh_env.event_logger import FieldTracker, format_events
from yugioh_env.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    SELECT_MSGS,
    MSG_WIN,
    PHASE_DRAW,
    PHASE_STANDBY,
    PHASE_MAIN1,
    PHASE_BATTLE_START,
    PHASE_BATTLE,
    PHASE_MAIN2,
    PHASE_END,
)
from yugioh_env.duel import Duel
from yugioh_env.lib_loader import load_library
from yugioh_env.models import YuGiOhAction, YuGiOhObservation, YuGiOhState
from yugioh_env.observation import build_observation
from yugioh_env.opponent import Opponent, RandomOpponent

logger = logging.getLogger(__name__)

PHASE_NAMES = {
    PHASE_DRAW: "draw",
    PHASE_STANDBY: "standby",
    PHASE_MAIN1: "main1",
    PHASE_BATTLE_START: "battle_start",
    PHASE_BATTLE: "battle",
    PHASE_MAIN2: "main2",
    PHASE_END: "end",
}


class YuGiOhEnvironment(Environment):
    """Server-side Yu-Gi-Oh! environment.

    Config keys:
        lib_path: Path to libocgcore shared library
        db_path: Path to cards.cdb
        script_dirs: List of paths to Lua script directories
        deck_path: Path to .ydk deck file (used for both players)
        deck0_path: Path to player 0 deck
        deck1_path: Path to player 1 deck
        opponent: Opponent spec — "random", "greedy", or "model:path/to/ckpt.pt"
        opponent_seed: Random seed for opponent
        opponent_device: Device for model opponent ("cpu" or "cuda", default "cpu")
        starting_lp: Starting life points (default 8000)
        agent_player: Which player the agent controls (0, 1, or "random").
                      Player 0 always goes first. Default 0.

    Environment variables (used as fallbacks when config keys are absent):
        YUGIOH_OPPONENT: Opponent spec (e.g. "greedy", "model:path/to/ckpt.pt")
        YUGIOH_OPPONENT_DEVICE: Device for model opponent (default "cpu")
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}

        # Resolve paths
        project_root = Path(__file__).resolve().parent.parent.parent
        lib_path = config.get("lib_path") or os.environ.get("YUGIOH_LIB_PATH")
        db_path = config.get("db_path") or os.environ.get(
            "YUGIOH_DB_PATH", str(project_root / "assets" / "cards.cdb")
        )
        script_dirs = config.get("script_dirs") or [
            project_root / "third_party" / "CardScripts" / "official",
            project_root / "third_party" / "CardScripts" / "pre-release",
            project_root / "third_party" / "CardScripts",
        ]
        deck_path = config.get("deck_path", str(project_root / "assets" / "decks" / "starter.ydk"))
        self._deck0_path = config.get("deck0_path", deck_path)
        self._deck1_path = config.get("deck1_path", deck_path)
        self._starting_lp = config.get("starting_lp", 8000)

        # Initialize components
        self._lib = load_library(lib_path)
        self._card_db = CardDatabase(db_path)
        self._script_dirs = [Path(d) for d in script_dirs]

        # Agent player: 0 = go first, 1 = go second, "random" = coin flip per episode
        agent_player_cfg = config.get("agent_player", 0)
        if agent_player_cfg not in (0, 1, "random"):
            raise ValueError(f"agent_player must be 0, 1, or 'random', got {agent_player_cfg!r}")
        self._agent_player_setting = agent_player_cfg
        self._agent_player = agent_player_cfg if isinstance(agent_player_cfg, int) else 0

        # Opponent — spec string: "random", "greedy", or "model:path/to/ckpt.pt"
        opponent_spec = config.get("opponent") or os.environ.get(
            "YUGIOH_OPPONENT", "random"
        )
        if opponent_spec.startswith("model:"):
            opponent_type = "model"
            opponent_checkpoint = opponent_spec[len("model:"):]
        else:
            opponent_type = opponent_spec
            opponent_checkpoint = ""

        opponent_seed = config.get("opponent_seed")
        if opponent_type == "model":
            from yugioh_env.opponent import ModelOpponent
            if not opponent_checkpoint:
                raise ValueError(
                    "model opponent requires a checkpoint path "
                    "(e.g. opponent='model:path/to/ckpt.pt')"
                )
            opponent_device = config.get("opponent_device") or os.environ.get(
                "YUGIOH_OPPONENT_DEVICE", "cpu"
            )
            self._opponent: Opponent = ModelOpponent(opponent_checkpoint, device=opponent_device)
        elif opponent_type == "greedy":
            from yugioh_env.opponent import GreedyOpponent
            self._opponent = GreedyOpponent()
        else:
            self._opponent = RandomOpponent(seed=opponent_seed)

        # Duel state
        self._duel: Duel | None = None
        self._mapper = ActionMapper()
        self._current_msg: dict | None = None
        self._episode_count = 0
        self._step_count = 0
        # Multi-step card selection accumulator (managed by environment)
        self._card_sel: list[int] = []
        # Persistent field tracker for event log card-code resolution
        self._field_tracker = FieldTracker()

    @staticmethod
    def _validate_deck(deck: dict, label: str) -> None:
        """Validate an inline deck dict.

        Raises ValueError if the deck is malformed.
        """
        if "main" not in deck:
            raise ValueError(f"{label}: missing 'main' key")
        main = deck["main"]
        if not isinstance(main, list) or len(main) < 40 or len(main) > 60:
            raise ValueError(
                f"{label}: main deck must have 40-60 cards, got {len(main) if isinstance(main, list) else type(main)}"
            )
        extra = deck.get("extra", [])
        if not isinstance(extra, list) or len(extra) > 15:
            raise ValueError(
                f"{label}: extra deck must have 0-15 cards, got {len(extra) if isinstance(extra, list) else type(extra)}"
            )
        for card in main + extra:
            if not isinstance(card, int) or card <= 0:
                raise ValueError(f"{label}: card codes must be positive integers, got {card!r}")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        deck0: Optional[dict[str, list[int]]] = None,
        deck1: Optional[dict[str, list[int]]] = None,
        agent_player: Optional[int | str] = None,
        **kwargs: Any,
    ) -> YuGiOhObservation:
        """Start a new duel and return the initial observation.

        Args:
            seed: RNG seed for this episode.
            episode_id: Optional episode identifier.
            deck0: Inline deck for engine player 0 ({"main": [...], "extra": [...]}).
                   Falls back to server-configured default if None.
                   Note: deck0/deck1 always map to engine player 0/1 (turn order),
                   not agent/opponent.
            deck1: Inline deck for engine player 1, same format as deck0.
            agent_player: Override which player the agent controls for this episode.
                          0 = go first, 1 = go second, "random" = coin flip.
                          If None, uses the value from config.
        """
        # Clean up previous duel
        if self._duel is not None:
            self._duel.destroy()

        self._episode_count += 1
        self._step_count = 0
        self._field_tracker.reset()
        duel_seed = seed if seed is not None else self._episode_count

        # Resolve agent player for this episode
        setting = agent_player if agent_player is not None else self._agent_player_setting
        if setting == "random":
            self._agent_player = stdlib_random.Random(duel_seed).randint(0, 1)
        else:
            self._agent_player = int(setting)

        # Re-seed opponent for reproducibility
        self._opponent.reseed(duel_seed)

        # Resolve decks: use provided inline dicts or fall back to configured paths
        if deck0 is not None:
            self._validate_deck(deck0, "deck0")
        if deck1 is not None:
            self._validate_deck(deck1, "deck1")

        effective_deck0 = deck0 if deck0 is not None else self._deck0_path
        effective_deck1 = deck1 if deck1 is not None else self._deck1_path

        # Create duel
        self._duel = Duel(self._lib, self._card_db, self._script_dirs)
        self._duel.create(
            deck0=effective_deck0,
            deck1=effective_deck1,
            seed=duel_seed,
            starting_lp=self._starting_lp,
        )

        # Process until agent's first choice
        return self._process_to_agent_choice()

    def step(
        self,
        action: YuGiOhAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> YuGiOhObservation:
        """Execute an action and return the resulting observation."""
        if self._duel is None or self._duel.is_finished:
            return self._make_terminal_observation()

        self._step_count += 1

        # Convert action to response
        try:
            response = self._mapper.action_to_response(action.action_index)
        except (ValueError, IndexError) as e:
            logger.error("Invalid action %d: %s", action.action_index, e)
            # Send first valid action as fallback
            if self._mapper.num_actions > 0:
                response = self._mapper.action_to_response(0)
            else:
                return self._make_terminal_observation()

        if response is None:
            # Multi-step: accumulate picked card, re-present with updated msg
            card_idx = self._mapper.get_action_index(action.action_index)
            self._card_sel.append(card_idx)
            updated_msg = {**self._current_msg, "_selected": list(self._card_sel)}
            self._mapper.update(updated_msg)
            return self._make_observation()

        self._card_sel.clear()
        self._duel.send_response(response)

        # Process until agent's next choice or game end
        return self._process_to_agent_choice()

    @property
    def state(self) -> YuGiOhState:
        """Return current episode metadata."""
        if self._duel is None:
            return YuGiOhState()

        gs = self._duel.game_state
        phase_name = PHASE_NAMES.get(gs.phase, "unknown")

        return YuGiOhState(
            step_count=self._step_count,
            turn_count=gs.turn_count,
            phase=phase_name,
            my_lp=gs.lp[self._agent_player],
            opp_lp=gs.lp[1 - self._agent_player],
            my_hand_count=gs.hand_count[self._agent_player],
            opp_hand_count=gs.hand_count[1 - self._agent_player],
        )

    def _process_to_agent_choice(self) -> YuGiOhObservation:
        """Process the duel, auto-play opponent turns, until agent must decide."""
        all_events: list[dict] = []
        while True:
            msg, gs, events = self._duel.process_until_choice()
            all_events.extend(events)

            if msg is None:
                # Game ended or error
                event_log = format_events(all_events, self._agent_player, self._card_db.get_card_name, self._field_tracker)
                return self._make_terminal_observation(event_log=event_log)

            msg_type = msg.get("msg_type")
            player = msg.get("player", -1)

            if player == self._agent_player and msg_type in SELECT_MSGS:
                # Agent's turn to decide
                self._current_msg = msg
                self._card_sel.clear()
                self._mapper.update(msg)
                event_log = format_events(all_events, self._agent_player, self._card_db.get_card_name, self._field_tracker)
                return self._make_observation(event_log=event_log)

            elif player != self._agent_player and msg_type in SELECT_MSGS:
                # Opponent's turn - auto-play (loop for multi-step selections)
                opp_mapper = ActionMapper()
                opp_mapper.update(msg)
                opp_sel: list[int] = []
                if opp_mapper.num_actions > 0:
                    response = None
                    while response is None and opp_mapper.num_actions > 0:
                        if self._opponent.needs_observation:
                            opp_obs = build_observation(
                                self._duel.game_state,
                                msg,
                                1 - self._agent_player,
                                query_fn=lambda p, l: self._duel.query_location(p, l),
                            )
                            opp_obs["actions"] = opp_mapper.get_action_features()
                            opp_obs["action_mask"] = opp_mapper.get_action_mask()
                            self._opponent.set_observation(opp_obs)
                        opp_action = self._opponent.select_action(msg, opp_mapper)
                        opp_action = min(opp_action, opp_mapper.num_actions - 1)
                        response = opp_mapper.action_to_response(opp_action)
                        if response is None:
                            opp_sel.append(opp_mapper.get_action_index(opp_action))
                            opp_mapper.update({**msg, "_selected": list(opp_sel)})
                    if response is not None:
                        self._duel.send_response(response)
                else:
                    logger.warning("Opponent has no actions for msg_type=%d", msg_type)
                    event_log = format_events(all_events, self._agent_player, self._card_db.get_card_name, self._field_tracker)
                    return self._make_terminal_observation(event_log=event_log)
            else:
                # Unknown message, try continuing
                event_log = format_events(all_events, self._agent_player, self._card_db.get_card_name, self._field_tracker)
                return self._make_terminal_observation(event_log=event_log)

    def _make_observation(self, event_log: list[str] | None = None) -> YuGiOhObservation:
        """Build observation from current state."""

        def query_fn(player: int, location: int) -> list[dict]:
            return self._duel.query_location(player, location)

        obs_data = build_observation(
            self._duel.game_state,
            self._current_msg,
            self._agent_player,
            query_fn=query_fn,
        )

        action_mask = self._mapper.get_action_mask()
        action_features = self._mapper.get_action_features()

        return YuGiOhObservation(
            cards=obs_data["cards"].tolist(),
            global_state=obs_data["global_state"].tolist(),
            actions=action_features.tolist(),
            action_mask=action_mask.tolist(),
            event_log=event_log or [],
            done=False,
            reward=0.0,
        )

    def _make_terminal_observation(self, event_log: list[str] | None = None) -> YuGiOhObservation:
        """Build terminal observation with reward."""
        reward = 0.0
        if self._duel and self._duel.game_state.is_finished:
            winner = self._duel.game_state.winner
            if winner == self._agent_player:
                reward = 1.0
            elif winner == 1 - self._agent_player:
                reward = -1.0
            # winner == 2 or other = draw = 0.0

        # Build a minimal observation
        obs_data = build_observation(
            self._duel.game_state if self._duel else None,
            None,
            self._agent_player,
        ) if self._duel else {"cards": [], "global_state": []}

        cards = obs_data["cards"].tolist() if hasattr(obs_data.get("cards", None), "tolist") else []
        global_state = obs_data["global_state"].tolist() if hasattr(obs_data.get("global_state", None), "tolist") else []

        return YuGiOhObservation(
            cards=cards,
            global_state=global_state,
            actions=[],
            action_mask=[0] * 32,
            event_log=event_log or [],
            done=True,
            reward=reward,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._duel is not None:
            self._duel.destroy()
            self._duel = None
        if self._card_db is not None:
            self._card_db.close()
