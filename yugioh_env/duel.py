"""High-level Duel class wrapping the OCG C API lifecycle."""

from __future__ import annotations

import ctypes
import logging
import struct
from pathlib import Path
from typing import Any

from yugioh_env.constants import (
    DUEL_MODE_MR5,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_REMOVED,
    OCG_DUEL_CREATION_SUCCESS,
    OCG_DUEL_STATUS_AWAITING,
    OCG_DUEL_STATUS_CONTINUE,
    OCG_DUEL_STATUS_END,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
    SELECT_MSGS,
    MSG_WIN,
    MSG_RETRY,
    QUERY_BASIC,
    TYPE_FUSION,
    TYPE_SYNCHRO,
    TYPE_XYZ,
    TYPE_LINK,
)
from yugioh_env.core_types import (
    OCG_DuelOptions,
    OCG_NewCardInfo,
    OCG_Player,
    OCG_QueryInfo,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from yugioh_env.callbacks import DuelCallbacks
from yugioh_env.card_database import CardDatabase
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.game_state import GameState
from yugioh_env.message_parser import parse_messages
from yugioh_env.observation import _parse_query_buffer

logger = logging.getLogger(__name__)

EXTRA_DECK_TYPES = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK


class Duel:
    """Manages a single duel instance.

    Usage:
        with Duel(lib, card_db, script_dirs) as duel:
            duel.create(deck0, deck1, seed=42)
            while not duel.is_finished:
                msg, state = duel.process_until_choice()
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
        if isinstance(deck0, (str, Path)):
            deck0 = parse_ydk(deck0)
        if isinstance(deck1, (str, Path)):
            deck1 = parse_ydk(deck1)

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
            startingLP=starting_lp,
            startingDrawCount=starting_draw,
            drawCountPerTurn=draw_per_turn,
        )
        options.team2 = OCG_Player(
            startingLP=starting_lp,
            startingDrawCount=starting_draw,
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

        # Add cards
        self._add_deck_cards(0, deck0)
        self._add_deck_cards(1, deck1)

        # Start duel
        self._lib.OCG_StartDuel(self._duel_handle)

        # Initialize game state counts from the engine (MSG_START may not be
        # emitted by the edo9300 fork, so query the engine directly).
        for p in range(2):
            self._game_state.deck_count[p] = self.query_count(p, LOCATION_DECK)
            self._game_state.extra_count[p] = self.query_count(p, LOCATION_EXTRA)
        self._game_state.lp = [starting_lp, starting_lp]

    def _add_deck_cards(self, team: int, deck: dict[str, list[int]]) -> None:
        """Add all cards from a deck to the duel."""
        # Main deck
        for seq, code in enumerate(deck.get("main", [])):
            card = self._card_db.get_card(code)
            card_type = card["type"] if card else 0
            # Check if card belongs in extra deck
            if card_type & EXTRA_DECK_TYPES:
                self._add_card(team, code, LOCATION_EXTRA, 0)
            else:
                self._add_card(team, code, LOCATION_DECK, seq)

        # Extra deck
        for seq, code in enumerate(deck.get("extra", [])):
            self._add_card(team, code, LOCATION_EXTRA, seq)

    def _add_card(self, team: int, code: int, location: int, seq: int) -> None:
        """Add a single card to the duel."""
        info = OCG_NewCardInfo()
        info.team = team
        info.duelist = 0
        info.code = code
        info.con = team
        info.loc = location
        info.seq = seq
        info.pos = POS_FACEDOWN_DEFENSE if location == LOCATION_EXTRA else POS_FACEDOWN_DEFENSE
        self._lib.OCG_DuelNewCard(self._duel_handle, ctypes.byref(info))

    def process_until_choice(self) -> tuple[dict | None, GameState]:
        """Process the duel until a player-choice message or game end.

        Returns:
            (select_msg, game_state) where select_msg is None if game ended.
        """
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
                        return None, self._game_state

                    if msg_type == MSG_RETRY:
                        logger.error("Got MSG_RETRY - last response was invalid!")
                        return None, self._game_state

                    if msg_type in SELECT_MSGS:
                        return msg, self._game_state

            if status == OCG_DUEL_STATUS_END:
                self._is_finished = True
                self._game_state.is_finished = True
                return None, self._game_state

            if status == OCG_DUEL_STATUS_AWAITING:
                # Should have gotten a SELECT message above
                # If we didn't, something went wrong
                logger.warning("AWAITING status but no SELECT message found")
                return None, self._game_state

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
            return _parse_query_buffer(buf)
        return []

    def query_card(self, player: int, location: int, sequence: int) -> dict | None:
        """Query a single card."""
        if self._duel_handle is None:
            return None
        info = OCG_QueryInfo()
        info.flags = QUERY_BASIC
        info.con = player
        info.loc = location
        info.seq = sequence
        info.overlay_seq = 0

        length = c_uint32()
        buf_ptr = self._lib.OCG_DuelQuery(
            self._duel_handle, ctypes.byref(length), ctypes.byref(info)
        )
        if length.value > 0 and buf_ptr:
            buf = ctypes.string_at(buf_ptr, length.value)
            cards = _parse_query_buffer(buf)
            return cards[0] if cards else None
        return None

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
        try:
            self.destroy()
        except Exception:
            pass
