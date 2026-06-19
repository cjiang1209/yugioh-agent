"""High-level Duel class wrapping the OCG C API lifecycle."""

from __future__ import annotations

import ctypes
import logging
import random
from contextlib import suppress
from pathlib import Path
from typing import Any

from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import (
    DUEL_MODE_MR5,
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_RETRY,
    MSG_WIN,
    OCG_DUEL_CREATION_SUCCESS,
    OCG_DUEL_STATUS_AWAITING,
    OCG_DUEL_STATUS_END,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    QUERY_BASIC,
    SELECT_MSGS,
)
from yugioh_core.query_buffer import parse_query_location
from yugioh_env.callbacks import DuelCallbacks
from yugioh_env.core_types import (
    OCG_DuelOptions,
    OCG_NewCardInfo,
    OCG_Player,
    OCG_QueryInfo,
    c_uint32,
    c_void_p,
)
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.game_state import GameState
from yugioh_env.message_parser import parse_messages
from yugioh_env.puzzle import generate_disable_lua, load_puzzle, validate_puzzle

logger = logging.getLogger(__name__)


class Duel:
    """Manages a single duel instance.

    Usage:
        with Duel(lib, card_db, script_dirs) as duel:
            duel.create(deck0, deck1, seed=42)
            while not duel.is_finished:
                msg, state, events = duel.process_until_choice()
                if msg is not None:
                    response = build_response(msg)
                    duel.send_response(response)
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        card_db: CardDatabase,
        script_dirs: list[Path],
    ):
        self._lib = lib
        self._card_db = card_db
        self._script_dirs = script_dirs
        self._duel_handle: int | None = None
        self._duel_ptr_holder: list[int | None] = [None]
        self._callbacks: DuelCallbacks | None = None
        self._game_state = GameState()
        self._is_finished = False

    @property
    def game_state(self) -> GameState:
        return self._game_state

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def create(
        self,
        deck0: dict[str, list[int]] | str | Path,
        deck1: dict[str, list[int]] | str | Path,
        seed: int = 0,
        starting_lp: int = 8000,
        starting_draw: int = 5,
        draw_per_turn: int = 1,
        flags: int = DUEL_MODE_MR5,
    ) -> None:
        """Create and start a new duel."""
        # Parse decks if given as paths
        if isinstance(deck0, str | Path):
            deck0 = parse_ydk(deck0)
        if isinstance(deck1, str | Path):
            deck1 = parse_ydk(deck1)

        self._init_duel(
            seed=seed,
            lp0=starting_lp,
            lp1=starting_lp,
            starting_draw0=starting_draw,
            starting_draw1=starting_draw,
            draw_per_turn=draw_per_turn,
            flags=flags,
        )

        # Add cards (shuffle main decks using the seed for determinism)
        rng = random.Random(seed)
        self._add_deck_cards(0, deck0, rng)
        self._add_deck_cards(1, deck1, rng)

        # Start duel
        self._lib.OCG_StartDuel(self._duel_handle)

        # Initialize game state counts from the engine (MSG_START may not be
        # emitted by the edo9300 fork, so query the engine directly).
        self._sync_zone_counts([starting_lp, starting_lp])

    def create_puzzle(
        self,
        state: dict | str | Path,
        seed: int = 0,
        flags: int = DUEL_MODE_MR5,
    ) -> None:
        """Create a duel from a puzzle state specification.

        *state* may be a validated dict, a raw dict (passed through
        ``validate_puzzle``), or a path to a JSON/YAML file (loaded via
        ``load_puzzle``).
        """
        # Parse / validate
        if isinstance(state, str | Path):
            state = load_puzzle(state)
        else:
            state = validate_puzzle(state)

        lp0 = state["player0"]["lp"]
        lp1 = state["player1"]["lp"]

        self._init_duel(
            seed=seed,
            lp0=lp0,
            lp1=lp1,
            starting_draw0=0,
            starting_draw1=0,
            draw_per_turn=1,
            flags=flags,
        )

        # Place cards for each player
        self._place_puzzle_cards(0, state["player0"])
        self._place_puzzle_cards(1, state["player1"])

        # Disable marked cards via Lua
        disable_lua = generate_disable_lua(state)
        if disable_lua is not None:
            content = disable_lua.encode("utf-8")
            ok = self._lib.OCG_LoadScript(
                self._duel_handle,
                content,
                len(content),
                b"puzzle_disable.lua",
            )
            if not ok:
                raise RuntimeError("Failed to load puzzle disable script")

        # Start duel (draws 0 cards)
        self._lib.OCG_StartDuel(self._duel_handle)

        # Initialize game state counts from the engine
        self._sync_zone_counts([lp0, lp1])

    def _sync_zone_counts(self, lp: list[int]) -> None:
        """Query the engine for all zone counts and set game state."""
        for p in range(2):
            self._game_state.deck_count[p] = self.query_count(p, LOCATION_DECK)
            self._game_state.extra_count[p] = self.query_count(p, LOCATION_EXTRA)
            self._game_state.hand_count[p] = self.query_count(p, LOCATION_HAND)
            self._game_state.mzone_count[p] = self.query_count(p, LOCATION_MZONE)
            self._game_state.szone_count[p] = self.query_count(p, LOCATION_SZONE)
            self._game_state.grave_count[p] = self.query_count(p, LOCATION_GRAVE)
            self._game_state.banished_count[p] = self.query_count(p, LOCATION_BANISHED)
        self._game_state.lp = lp

    def _init_duel(
        self,
        seed: int,
        lp0: int,
        lp1: int,
        starting_draw0: int,
        starting_draw1: int,
        draw_per_turn: int,
        flags: int,
    ) -> None:
        """Shared duel-creation setup: callbacks, engine init, startup scripts."""
        # Reset state
        self._game_state.reset()
        self._is_finished = False

        # Set up callbacks
        self._callbacks = DuelCallbacks(
            card_db=self._card_db,
            script_dirs=self._script_dirs,
            duel_ptr_holder=self._duel_ptr_holder,
            lib=self._lib,
        )

        # Build duel options
        options = OCG_DuelOptions()
        # seed[4] must not be all-zero (OCG_DUEL_CREATION_NULL_RNG_SEED).
        # Spread the seed across all 4 uint64 slots using simple mixing.
        s = seed if seed != 0 else 1
        options.seed[0] = s & 0xFFFFFFFFFFFFFFFF
        options.seed[1] = ((s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF) or 1
        options.seed[2] = ((s * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF) or 1
        options.seed[3] = ((s ^ 0xDEADBEEFCAFEBABE) & 0xFFFFFFFFFFFFFFFF) or 1
        options.flags = flags
        options.team1 = OCG_Player(
            startingLP=lp0,
            startingDrawCount=starting_draw0,
            drawCountPerTurn=draw_per_turn,
        )
        options.team2 = OCG_Player(
            startingLP=lp1,
            startingDrawCount=starting_draw1,
            drawCountPerTurn=draw_per_turn,
        )
        options.cardReader = self._callbacks.card_reader_cb
        options.payload1 = None
        options.scriptReader = self._callbacks.script_reader_cb
        options.payload2 = None
        options.logHandler = self._callbacks.log_handler_cb
        options.payload3 = None
        options.cardReaderDone = self._callbacks.card_reader_done_cb
        options.payload4 = None
        options.enableUnsafeLibraries = 0

        # Create duel
        duel_ptr = c_void_p()
        status = self._lib.OCG_CreateDuel(ctypes.byref(duel_ptr), ctypes.byref(options))
        if status != OCG_DUEL_CREATION_SUCCESS:
            raise RuntimeError(f"OCG_CreateDuel failed with status {status}")

        self._duel_handle = duel_ptr.value
        self._duel_ptr_holder[0] = self._duel_handle

        # Load core Lua scripts (constants, utility functions, procs) that
        # card scripts depend on.  The engine only requests individual card
        # scripts via the script_reader callback; the host must pre-load
        # the shared runtime scripts itself.
        self._load_startup_scripts()

    # Zone key → (location constant, default position) for simple card-list zones.
    _LIST_ZONE_MAP: list[tuple[str, int, int]] = [
        ("hand", LOCATION_HAND, POS_FACEDOWN_DEFENSE),
        ("grave", LOCATION_GRAVE, POS_FACEUP_ATTACK),
        ("banished", LOCATION_BANISHED, POS_FACEUP_ATTACK),
        ("deck", LOCATION_DECK, POS_FACEDOWN_DEFENSE),
        ("extra", LOCATION_EXTRA, POS_FACEDOWN_DEFENSE),
    ]

    def _place_puzzle_cards(self, player: int, config: dict) -> None:
        """Place cards for a single player from a puzzle configuration."""
        # Simple list zones (card codes or dicts with code+disabled, implicit position)
        for key, location, default_pos in self._LIST_ZONE_MAP:
            for seq, entry in enumerate(config.get(key, [])):
                code = entry["code"] if isinstance(entry, dict) else entry
                self._add_card(player, code, location, seq, pos=default_pos)

        # Field zones (dict entries with explicit position and sequence)
        for entry in config.get("monster_zone", []):
            self._add_card(
                player,
                entry["code"],
                LOCATION_MZONE,
                entry["seq"],
                pos=entry["pos"],
            )
        for entry in config.get("spell_zone", []):
            self._add_card(
                player,
                entry["code"],
                LOCATION_SZONE,
                entry["seq"],
                pos=entry["pos"],
            )

    # Scripts that must be loaded (in order) before any card scripts run.
    # constant.lua defines numeric constants; utility.lua defines GetID()
    # and the Auxiliary table; the remaining files extend Auxiliary.
    _STARTUP_SCRIPTS: list[str] = [
        "constant.lua",
        "utility.lua",
        "archetype_setcode_constants.lua",
        "card_counter_constants.lua",
        "proc_normal.lua",
        "proc_fusion.lua",
        "proc_fusion_spell.lua",
        "proc_ritual.lua",
        "proc_synchro.lua",
        "proc_xyz.lua",
        "proc_pendulum.lua",
        "proc_link.lua",
        "proc_equip.lua",
        "proc_gemini.lua",
        "proc_spirit.lua",
        "proc_union.lua",
        "proc_maximum.lua",
        "proc_rush.lua",
        "proc_skill.lua",
        "proc_persistent.lua",
        "proc_workaround.lua",
        "cards_specific_functions.lua",
        "deprecated_functions.lua",
    ]

    def _load_startup_scripts(self) -> None:
        """Load core Lua runtime scripts into the duel engine.

        These define constants and helper functions (e.g. ``GetID()``) that
        every individual card script relies on.
        """
        for name in self._STARTUP_SCRIPTS:
            for script_dir in self._script_dirs:
                path = script_dir / name
                if path.exists():
                    content = path.read_bytes()
                    ok = self._lib.OCG_LoadScript(
                        self._duel_handle,
                        content,
                        len(content),
                        name.encode("utf-8"),
                    )
                    if not ok:
                        logger.warning("Failed to load startup script: %s", name)
                    break
            else:
                logger.debug("Startup script not found: %s", name)

    def _add_deck_cards(self, team: int, deck: dict[str, list[int]], rng: random.Random) -> None:
        """Add all cards from a deck to the duel.

        The main deck is shuffled using *rng* before insertion because the
        engine's Startup processor clears shuffle flags before the opening draw.
        """
        # Shuffle main deck, then add cards
        main_codes = list(deck.get("main", []))
        rng.shuffle(main_codes)
        for seq, code in enumerate(main_codes):
            self._add_card(team, code, LOCATION_DECK, seq)

        # Extra deck
        for seq, code in enumerate(deck.get("extra", [])):
            self._add_card(team, code, LOCATION_EXTRA, seq)

    def _add_card(
        self, team: int, code: int, location: int, seq: int, *, pos: int = POS_FACEDOWN_DEFENSE
    ) -> None:
        """Add a single card to the duel."""
        info = OCG_NewCardInfo()
        info.team = team
        info.duelist = 0
        info.code = code
        info.con = team
        info.loc = location
        info.seq = seq
        info.pos = pos
        self._lib.OCG_DuelNewCard(self._duel_handle, ctypes.byref(info))

    def process_until_choice(self) -> tuple[dict | None, GameState, list[dict]]:
        """Process the duel until a player-choice message or game end.

        Returns:
            (select_msg, game_state, info_msgs) where select_msg is None if
            game ended, and info_msgs is a list of informational message dicts
            encountered before the choice point.
        """
        info_msgs: list[dict] = []
        while True:
            status = self._lib.OCG_DuelProcess(self._duel_handle)

            # Get messages
            length = c_uint32()
            buf_ptr = self._lib.OCG_DuelGetMessage(self._duel_handle, ctypes.byref(length))
            if length.value > 0 and buf_ptr:
                buf = ctypes.string_at(buf_ptr, length.value)
                messages = parse_messages(buf)

                for msg in messages:
                    self._game_state.update(msg)
                    msg_type = msg.get("msg_type")

                    if msg_type == MSG_WIN:
                        self._is_finished = True
                        self._game_state.is_finished = True
                        self._game_state.winner = msg.get("player", -1)
                        return None, self._game_state, info_msgs

                    if msg_type == MSG_RETRY:
                        logger.error("Got MSG_RETRY - last response was invalid!")
                        return None, self._game_state, info_msgs

                    if msg_type in SELECT_MSGS:
                        return msg, self._game_state, info_msgs

                    info_msgs.append(msg)

            if status == OCG_DUEL_STATUS_END:
                self._is_finished = True
                self._game_state.is_finished = True
                return None, self._game_state, info_msgs

            if status == OCG_DUEL_STATUS_AWAITING:
                # Should have gotten a SELECT message above
                # If we didn't, something went wrong
                logger.warning("AWAITING status but no SELECT message found")
                return None, self._game_state, info_msgs

    def send_response(self, response: bytes) -> None:
        """Send a response buffer to the engine."""
        self._lib.OCG_DuelSetResponse(
            self._duel_handle,
            response,
            len(response),
        )

    def query_location(self, player: int, location: int) -> list[dict]:
        """Query all cards at a location for a player."""
        if self._duel_handle is None:
            return []
        info = OCG_QueryInfo()
        info.flags = QUERY_BASIC
        info.con = player
        info.loc = location
        info.seq = 0
        info.overlay_seq = 0

        length = c_uint32()
        buf_ptr = self._lib.OCG_DuelQueryLocation(
            self._duel_handle, ctypes.byref(length), ctypes.byref(info)
        )
        if length.value > 0 and buf_ptr:
            buf = ctypes.string_at(buf_ptr, length.value)
            return parse_query_location(buf)
        return []

    def query_count(self, player: int, location: int) -> int:
        """Query the number of cards at a location."""
        if self._duel_handle is None:
            return 0
        return self._lib.OCG_DuelQueryCount(self._duel_handle, player, location)

    def destroy(self) -> None:
        """Destroy the duel and free resources."""
        if self._duel_handle is not None:
            self._lib.OCG_DestroyDuel(self._duel_handle)
            self._duel_handle = None
            self._duel_ptr_holder[0] = None
        if self._callbacks:
            self._callbacks.reset()

    def __enter__(self) -> Duel:
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        with suppress(Exception):
            self.destroy()
