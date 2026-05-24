"""Load libocgcore shared library and set up function signatures."""

import ctypes
import platform
from ctypes import (
    POINTER,
    c_char_p,
    c_int,
    c_uint8,
    c_uint32,
    c_void_p,
)
from pathlib import Path

from yugioh_env.core_types import (
    OCG_Duel,
    OCG_DuelOptions,
    OCG_NewCardInfo,
    OCG_QueryInfo,
)


def _find_library() -> Path:
    """Find the built libocgcore shared library."""
    ext = "dylib" if platform.system() == "Darwin" else "so"
    # Check common locations relative to this file
    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir.parent / "build" / f"libocgcore.{ext}",
        this_dir / f"libocgcore.{ext}",
        Path(f"build/libocgcore.{ext}"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"libocgcore.{ext} not found. Run: make build\nSearched: {[str(c) for c in candidates]}"
    )


def load_library(path: str | Path | None = None) -> ctypes.CDLL:
    """Load the OCG core library and configure all function signatures.

    Args:
        path: Explicit path to libocgcore. If None, auto-detect.

    Returns:
        Configured ctypes.CDLL instance.
    """
    if path is None:
        path = _find_library()
    lib = ctypes.cdll.LoadLibrary(str(path))
    _configure_signatures(lib)
    return lib


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes and restype for all OCG API functions."""

    # void OCG_GetVersion(int* major, int* minor)
    lib.OCG_GetVersion.argtypes = [POINTER(c_int), POINTER(c_int)]
    lib.OCG_GetVersion.restype = None

    # int OCG_CreateDuel(OCG_Duel* out, const OCG_DuelOptions* options)
    lib.OCG_CreateDuel.argtypes = [POINTER(c_void_p), POINTER(OCG_DuelOptions)]
    lib.OCG_CreateDuel.restype = c_int

    # void OCG_DestroyDuel(OCG_Duel duel)
    lib.OCG_DestroyDuel.argtypes = [OCG_Duel]
    lib.OCG_DestroyDuel.restype = None

    # void OCG_DuelNewCard(OCG_Duel duel, const OCG_NewCardInfo* info)
    lib.OCG_DuelNewCard.argtypes = [OCG_Duel, POINTER(OCG_NewCardInfo)]
    lib.OCG_DuelNewCard.restype = None

    # void OCG_StartDuel(OCG_Duel duel)
    lib.OCG_StartDuel.argtypes = [OCG_Duel]
    lib.OCG_StartDuel.restype = None

    # int OCG_DuelProcess(OCG_Duel duel)
    lib.OCG_DuelProcess.argtypes = [OCG_Duel]
    lib.OCG_DuelProcess.restype = c_int

    # void* OCG_DuelGetMessage(OCG_Duel duel, uint32_t* length)
    lib.OCG_DuelGetMessage.argtypes = [OCG_Duel, POINTER(c_uint32)]
    lib.OCG_DuelGetMessage.restype = c_void_p

    # void OCG_DuelSetResponse(OCG_Duel duel, const void* buffer, uint32_t length)
    lib.OCG_DuelSetResponse.argtypes = [OCG_Duel, c_void_p, c_uint32]
    lib.OCG_DuelSetResponse.restype = None

    # int OCG_LoadScript(OCG_Duel duel, const char* buffer, uint32_t length, const char* name)
    lib.OCG_LoadScript.argtypes = [OCG_Duel, c_char_p, c_uint32, c_char_p]
    lib.OCG_LoadScript.restype = c_int

    # uint32_t OCG_DuelQueryCount(OCG_Duel duel, uint8_t team, uint32_t loc)
    lib.OCG_DuelQueryCount.argtypes = [OCG_Duel, c_uint8, c_uint32]
    lib.OCG_DuelQueryCount.restype = c_uint32

    # void* OCG_DuelQuery(OCG_Duel duel, uint32_t* length, const OCG_QueryInfo* info)
    lib.OCG_DuelQuery.argtypes = [OCG_Duel, POINTER(c_uint32), POINTER(OCG_QueryInfo)]
    lib.OCG_DuelQuery.restype = c_void_p

    # void* OCG_DuelQueryLocation(OCG_Duel duel, uint32_t* length, const OCG_QueryInfo* info)
    lib.OCG_DuelQueryLocation.argtypes = [
        OCG_Duel,
        POINTER(c_uint32),
        POINTER(OCG_QueryInfo),
    ]
    lib.OCG_DuelQueryLocation.restype = c_void_p

    # void* OCG_DuelQueryField(OCG_Duel duel, uint32_t* length)
    lib.OCG_DuelQueryField.argtypes = [OCG_Duel, POINTER(c_uint32)]
    lib.OCG_DuelQueryField.restype = c_void_p
