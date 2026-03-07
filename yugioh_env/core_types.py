"""ctypes structures matching ygopro-core ocgapi_types.h."""

import ctypes
from ctypes import (
    Structure,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_int32,
    c_void_p,
    c_char_p,
    c_int,
    POINTER,
    CFUNCTYPE,
)

# ─── OCG_Duel is an opaque void* ────────────────────────────────────────────
OCG_Duel = c_void_p


# ─── Structures ──────────────────────────────────────────────────────────────
class OCG_CardData(Structure):
    _fields_ = [
        ("code", c_uint32),
        ("alias", c_uint32),
        ("setcodes", POINTER(c_uint16)),
        ("type", c_uint32),
        ("level", c_uint32),
        ("attribute", c_uint32),
        ("race", c_uint64),
        ("attack", c_int32),
        ("defense", c_int32),
        ("lscale", c_uint32),
        ("rscale", c_uint32),
        ("link_marker", c_uint32),
    ]


class OCG_Player(Structure):
    _fields_ = [
        ("startingLP", c_uint32),
        ("startingDrawCount", c_uint32),
        ("drawCountPerTurn", c_uint32),
    ]


# ─── Callback function types ────────────────────────────────────────────────
# void (*OCG_DataReader)(void* payload, uint32_t code, OCG_CardData* data)
OCG_DataReader = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(OCG_CardData))

# void (*OCG_DataReaderDone)(void* payload, OCG_CardData* data)
OCG_DataReaderDone = CFUNCTYPE(None, c_void_p, POINTER(OCG_CardData))

# int (*OCG_ScriptReader)(void* payload, OCG_Duel duel, const char* name)
OCG_ScriptReader = CFUNCTYPE(c_int, c_void_p, OCG_Duel, c_char_p)

# void (*OCG_LogHandler)(void* payload, const char* string, int type)
OCG_LogHandler = CFUNCTYPE(None, c_void_p, c_char_p, c_int)


class OCG_DuelOptions(Structure):
    _fields_ = [
        ("seed", c_uint64 * 4),
        ("flags", c_uint64),
        ("team1", OCG_Player),
        ("team2", OCG_Player),
        ("cardReader", OCG_DataReader),
        ("payload1", c_void_p),
        ("scriptReader", OCG_ScriptReader),
        ("payload2", c_void_p),
        ("logHandler", OCG_LogHandler),
        ("payload3", c_void_p),
        ("cardReaderDone", OCG_DataReaderDone),
        ("payload4", c_void_p),
        ("enableUnsafeLibraries", c_uint8),
    ]


class OCG_NewCardInfo(Structure):
    _fields_ = [
        ("team", c_uint8),
        ("duelist", c_uint8),
        ("code", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("pos", c_uint32),
    ]


class OCG_QueryInfo(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("overlay_seq", c_uint32),
    ]
