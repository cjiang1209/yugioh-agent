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
from yugioh_core.card_database import CardDatabase
from yugioh_core.encoding import MAX_ACTIONS
from yugioh_env.event_logger import FieldTracker, format_events
from yugioh_core.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    PHASE_BATTLE,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_DRAW,
    PHASE_END,
    PHASE_MAIN1,
    PHASE_MAIN2,
    PHASE_STANDBY,
    SELECT_MSGS,
    MSG_WIN,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,
    MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_POSITION,
    MSG_SELECT_PLACE,
    MSG_SELECT_DISFIELD,
)
from yugioh_env.duel import Duel
from yugioh_env.lib_loader import load_library
from yugioh_env.models import YuGiOhAction, YuGiOhObservation, YuGiOhState, ActionMeta
from yugioh_env.observation import build_observation
from yugioh_env.opponent import Opponent, make_opponent
from yugioh_env.server.board_state import build_board_state

logger = logging.getLogger(__name__)

# Lowercase phase names for the HTTP API response
_API_PHASE_NAMES = {
    PHASE_DRAW: "draw",
    PHASE_STANDBY: "standby",
    PHASE_MAIN1: "main1",
    PHASE_BATTLE_START: "battle_start",
    PHASE_BATTLE_STEP: "battle_step",
    PHASE_DAMAGE: "damage",
    PHASE_DAMAGE_CAL: "damage_calc",
    PHASE_BATTLE: "battle",
    PHASE_MAIN2: "main2",
    PHASE_END: "end",
}


def _resolve_opponent_device(config: dict[str, Any]) -> str:
    """Resolve the device for a model opponent.

    Config key wins over the ``YUGIOH_OPPONENT_DEVICE`` env var, default ``"cpu"``.
    Extracted so it can be unit-tested without booting the engine.
    """
    return config.get("opponent_device") or os.environ.get(
        "YUGIOH_OPPONENT_DEVICE", "cpu"
    )


def _build_action_meta_list(actions: list[dict]) -> list[ActionMeta | None]:
    """Build a length-MAX_ACTIONS list parallel to actions[], None for slots without meta.

    Validates each meta dict through Pydantic (raises ValidationError on bad kind)."""
    out: list[ActionMeta | None] = [None] * MAX_ACTIONS
    for i, action in enumerate(actions[:MAX_ACTIONS]):
        meta = action.get("meta")
        if meta is not None:
            out[i] = ActionMeta(**meta)
    return out


def _build_prompt_meta(mapper) -> dict | None:
    """Build prompt-level metadata from the mapper's current message.

    Mirrors the field-extraction logic of the previous server-side
    `describe_prompt`, minus the card_name lookup (which moves to
    ActionDescriber). Returns None when no active prompt.

    The returned dict always includes a documented `msg_type` field
    (the raw ygopro-core MSG_SELECT_* integer). It is part of the wire
    contract for `prompt_meta`: openenv HTTP clients receive it via
    `model_dump()`, and `ActionDescriber.describe_prompt` consumes
    (pops) it to derive the human-readable `type` enum.
    """
    msg_type = mapper.msg_type
    msg = mapper.msg
    if msg_type is None or not msg:
        return None
    result: dict = {"msg_type": msg_type}
    if msg_type == MSG_SELECT_EFFECTYN:
        result["card_code"] = msg.get("code", 0)
        result["location"] = msg.get("location", 0)
        result["desc"] = msg.get("desc", 0)
    elif msg_type == MSG_SELECT_YESNO:
        result["desc"] = msg.get("desc", 0)
    elif msg_type == MSG_SELECT_CARD:
        result["min"] = msg.get("min", 1)
        result["max"] = msg.get("max", 1)
        result["cancelable"] = bool(msg.get("cancelable", 0))
        result["selected_count"] = len(msg.get("_selected", []))
    elif msg_type == MSG_SELECT_TRIBUTE:
        selected = msg.get("_selected", [])
        cards = msg.get("cards", [])
        result["min_release"] = msg.get("min", 1)
        result["max_cards"] = msg.get("max", 1)
        result["cancelable"] = bool(msg.get("cancelable", 0))
        result["release_total"] = sum(
            cards[i].get("release_param", 1) for i in selected if i < len(cards)
        )
        result["cards_selected"] = len(selected)
    elif msg_type == MSG_SELECT_UNSELECT_CARD:
        result["min"] = msg.get("min", 1)
        result["max"] = msg.get("max", 1)
        result["finishable"] = bool(msg.get("finishable", 0))
    elif msg_type == MSG_SELECT_CHAIN:
        result["forced"] = bool(msg.get("forced", 0))
    elif msg_type == MSG_SELECT_POSITION:
        result["card_code"] = msg.get("code", 0)
    elif msg_type in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        result["count"] = msg.get("count", 1)
    return result


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
        deck_path = config.get("deck_path", str(project_root / "assets" / "decks" / "blue_eyes.ydk"))
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
        opponent_seed = config.get("opponent_seed")
        opponent_device = _resolve_opponent_device(config)
        self._opponent: Opponent = make_opponent(
            opponent_spec, seed=opponent_seed, device=opponent_device
        )

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
        # Intermediate board snapshots captured during _process_to_agent_choice()
        self._last_frames: list[dict] = []
        # When True, board snapshots include unhidden opponent card data
        self._open_cards: bool = False

    def set_opponent(self, opponent: Opponent) -> None:
        """Replace the opponent for subsequent episodes."""
        self._opponent = opponent

    @property
    def last_frames(self) -> list[dict]:
        """Intermediate board snapshots captured during the last process cycle."""
        return self._last_frames

    @property
    def current_msg(self) -> dict | None:
        """Active MSG_SELECT_* prompt, or None between prompts.

        In multi-step card selection this carries the accumulated `_selected`
        indices so readers see the narrowed prompt, not the original.
        """
        return self._current_msg

    @property
    def num_actions(self) -> int:
        """Legal action count for the active prompt (0 when none)."""
        if self._current_msg is None:
            return 0
        return self._mapper.num_actions

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
        open_cards: bool = False,
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
            open_cards: When True, board snapshots include full opponent card
                        data (unhidden) in the ``opponent`` dict (UI-only,
                        does not affect game logic).  Defaults to False.
        """
        # Apply open_cards before processing so frames include the data
        self._open_cards = open_cards

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

        self._last_frames = []
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
            # Multi-step: accumulate picked card, re-present with updated msg.
            # _current_msg must track the mapper update so current_msg/num_actions stay consistent.
            card_idx = self._mapper.get_action_index(action.action_index)
            self._card_sel.append(card_idx)
            updated_msg = {
                **self._current_msg,
                "_selected": list(self._card_sel),
                "_agent_player": self._agent_player,
            }
            self._mapper.update(updated_msg)
            self._current_msg = updated_msg
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
        phase_name = _API_PHASE_NAMES.get(gs.phase, "unknown")

        return YuGiOhState(
            step_count=self._step_count,
            turn_count=gs.turn_count,
            phase=phase_name,
            my_lp=gs.lp[self._agent_player],
            opp_lp=gs.lp[1 - self._agent_player],
            my_hand_count=gs.hand_count[self._agent_player],
            opp_hand_count=gs.hand_count[1 - self._agent_player],
        )

    def _build_game_state_dict(self) -> dict:
        """Build a game_state dict from the current duel state."""
        gs = self._duel.game_state if self._duel else None
        if gs is None:
            return {"turn": 0, "phase": "unknown", "is_my_turn": False, "chain_count": 0}
        return {
            "turn": gs.turn_count,
            "phase": _API_PHASE_NAMES.get(gs.phase, "unknown"),
            "is_my_turn": gs.current_player == self._agent_player,
            "chain_count": gs.chain_count,
        }

    def _capture_frame(self, events: list[dict]) -> None:
        """Format events and snapshot the board into a frame."""
        if not events:
            return
        chunk_log = format_events(
            events, self._agent_player,
            self._card_db.get_card_name, self._field_tracker,
        )
        if chunk_log:
            self._last_frames.append({
                "events": chunk_log,
                "board": build_board_state(self, open_cards=self._open_cards),
                "game_state": self._build_game_state_dict(),
            })

    def _flatten_frame_events(self) -> list[str]:
        """Collect all formatted event strings from captured frames."""
        return [e for f in self._last_frames for e in f["events"]]

    def _process_to_agent_choice(self) -> YuGiOhObservation:
        """Process the duel, auto-play opponent turns, until agent must decide."""
        self._last_frames = []
        while True:
            msg, gs, events = self._duel.process_until_choice()

            # Capture frame from this chunk's events
            self._capture_frame(events)

            if msg is None:
                # Game ended or error
                return self._make_terminal_observation(
                    event_log=self._flatten_frame_events(),
                )

            msg_type = msg.get("msg_type")
            player = msg.get("player", -1)

            if player == self._agent_player and msg_type in SELECT_MSGS:
                # Agent's turn to decide
                self._current_msg = msg
                self._card_sel.clear()
                self._mapper.update({**msg, "_agent_player": self._agent_player})
                return self._make_observation(
                    event_log=self._flatten_frame_events(),
                )

            elif player != self._agent_player and msg_type in SELECT_MSGS:
                # Opponent's turn - auto-play (loop for multi-step selections)
                opp_mapper = ActionMapper()
                opp_agent_player = 1 - self._agent_player
                opp_mapper.update({**msg, "_agent_player": opp_agent_player})
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
                        opp_action = self._opponent.select_action(msg, opp_mapper.num_actions)
                        opp_action = min(opp_action, opp_mapper.num_actions - 1)
                        response = opp_mapper.action_to_response(opp_action)
                        if response is None:
                            opp_sel.append(opp_mapper.get_action_index(opp_action))
                            opp_mapper.update({
                                **msg,
                                "_selected": list(opp_sel),
                                "_agent_player": opp_agent_player,
                            })
                    if response is not None:
                        self._duel.send_response(response)
                else:
                    logger.warning("Opponent has no actions for msg_type=%d", msg_type)
                    return self._make_terminal_observation(
                        event_log=self._flatten_frame_events(),
                    )
            else:
                # Unknown message, try continuing
                return self._make_terminal_observation(
                    event_log=self._flatten_frame_events(),
                )

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
        action_meta = _build_action_meta_list(self._mapper.actions)

        assert len(action_meta) == len(action_features) == len(action_mask), (
            f"action_meta/features/mask length drift: "
            f"{len(action_meta)}/{len(action_features)}/{len(action_mask)}"
        )

        return YuGiOhObservation(
            cards=obs_data["cards"].tolist(),
            global_state=obs_data["global_state"].tolist(),
            actions=action_features.tolist(),
            action_mask=action_mask.tolist(),
            action_meta=action_meta,
            prompt_meta=_build_prompt_meta(self._mapper),
            event_log=event_log or [],
            done=False,
            reward=0.0,
        )

    def _make_terminal_observation(self, event_log: list[str] | None = None) -> YuGiOhObservation:
        """Build terminal observation with reward."""
        # No active prompt past this point; keeps current_msg/num_actions in sync with action_mask=0.
        self._current_msg = None
        self._card_sel.clear()

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
            action_mask=[],
            action_meta=[],
            prompt_meta=None,
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
