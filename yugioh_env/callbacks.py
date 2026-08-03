"""Python callbacks for ygopro-core OCG_DuelOptions.

CRITICAL: All callback objects and any ctypes arrays they allocate must be
stored as instance attributes to prevent garbage collection (dangling pointer crash).
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from yugioh_core.constants import split_setcodes
from yugioh_env.core_types import (
    POINTER,
    OCG_CardData,
    OCG_DataReader,
    OCG_DataReaderDone,
    OCG_LogHandler,
    OCG_ScriptReader,
    c_char_p,
    c_uint16,
    c_void_p,
)

if TYPE_CHECKING:
    from yugioh_core.card_database import CardDatabase

logger = logging.getLogger(__name__)


class DuelCallbacks:
    """Manages callbacks and their prevent-GC storage for one duel.

    Attributes:
        card_reader_cb: ctypes callback for reading card data
        card_reader_done_cb: ctypes callback for cleanup after card read
        script_reader_cb: ctypes callback for loading Lua scripts
        log_handler_cb: ctypes callback for log messages
    """

    def __init__(
        self,
        card_db: CardDatabase,
        script_dirs: list[Path],
        duel_ptr_holder: list[int | None] | None = None,
        lib: ctypes.CDLL | None = None,
    ):
        self._card_db = card_db
        self._script_dirs = script_dirs
        self._lib = lib
        self._duel_ptr_holder = duel_ptr_holder or [None]
        # Storage for setcode arrays to prevent GC during card data lifetime
        self._setcode_storage: dict[int, ctypes.Array] = {}
        # Storage for loaded script content
        self._loaded_scripts: set[str] = set()

        # Create and store the callbacks
        self.card_reader_cb = OCG_DataReader(self._card_reader)
        self.card_reader_done_cb = OCG_DataReaderDone(self._card_reader_done)
        self.script_reader_cb = OCG_ScriptReader(self._script_reader)
        self.log_handler_cb = OCG_LogHandler(self._log_handler)

    def _card_reader(self, payload: c_void_p, code: int, data_ptr: POINTER(OCG_CardData)) -> None:
        """Fill OCG_CardData struct from card database."""
        card = self._card_db.get_card(code)
        data = data_ptr[0]
        if card is None:
            data.code = code
            data.alias = 0
            data.setcodes = ctypes.cast(ctypes.pointer(c_uint16(0)), POINTER(c_uint16))
            data.type = 0
            data.level = 0
            data.attribute = 0
            data.race = 0
            data.attack = 0
            data.defense = 0
            data.lscale = 0
            data.rscale = 0
            data.link_marker = 0
            return

        data.code = card["code"]
        data.alias = card.get("alias", 0)

        # Build null-terminated setcodes array and store to prevent GC
        setcodes = card.get("setcodes", [])
        if isinstance(setcodes, int):
            setcodes = split_setcodes(setcodes)
        arr = (c_uint16 * (len(setcodes) + 1))()
        for i, sc in enumerate(setcodes):
            arr[i] = sc
        arr[len(setcodes)] = 0
        self._setcode_storage[code] = arr
        data.setcodes = ctypes.cast(arr, POINTER(c_uint16))

        data.type = card.get("type", 0)
        data.level = card.get("level", 0)
        data.attribute = card.get("attribute", 0)
        data.race = card.get("race", 0)
        data.attack = card.get("attack", 0)
        # OCG_CardData is a ctypes struct, so None (Link monsters) needs a number.
        data.defense = card.get("defense") or 0
        data.lscale = card.get("lscale", 0)
        data.rscale = card.get("rscale", 0)
        data.link_marker = card.get("link_marker", 0)

    def _card_reader_done(self, payload: c_void_p, data_ptr: POINTER(OCG_CardData)) -> None:
        """Cleanup after card data has been consumed by the engine."""
        code = data_ptr[0].code
        self._setcode_storage.pop(code, None)

    def _script_reader(self, payload: c_void_p, duel: c_void_p, name: c_char_p) -> int:
        """Load a Lua script file into the duel engine."""
        script_name = name.decode("utf-8") if isinstance(name, bytes) else name
        if script_name in self._loaded_scripts:
            return 1

        for script_dir in self._script_dirs:
            script_path = script_dir / script_name
            if script_path.exists():
                content = script_path.read_bytes()
                duel_handle = self._duel_ptr_holder[0]
                if duel_handle is not None and self._lib is not None:
                    result = self._lib.OCG_LoadScript(
                        duel_handle,
                        content,
                        len(content),
                        name if isinstance(name, bytes) else name.encode("utf-8"),
                    )
                    if result:
                        self._loaded_scripts.add(script_name)
                    return result
                return 0
        logger.debug("Script not found: %s", script_name)
        return 0

    def _log_handler(self, payload: c_void_p, string: c_char_p, log_type: int) -> None:
        """Handle log messages from the engine."""
        msg = string.decode("utf-8", errors="replace") if isinstance(string, bytes) else str(string)
        if log_type == 0:  # OCG_LOG_TYPE_ERROR
            logger.error("OCG: %s", msg)
        elif log_type == 1:  # OCG_LOG_TYPE_FROM_SCRIPT
            logger.info("OCG script: %s", msg)
        else:
            logger.debug("OCG: %s", msg)

    def reset(self) -> None:
        """Reset state between duels."""
        self._setcode_storage.clear()
        self._loaded_scripts.clear()
