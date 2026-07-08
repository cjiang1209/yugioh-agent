"""Rolling event-history buffer feeding the CNN event branch.

Populated from the ENRICHED message stream (both players) inside the env.
Stores raw engine controller/turn_player; relativization to agent-relative
happens in ``to_tensor`` where ``agent_player`` is known.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from yugioh_core.constants import (
    HINT_ATTRIB,
    HINT_CODE,
    HINT_NUMBER,
    HINT_RACE,
    MSG_ATTACK,
    MSG_CHAINING,
    MSG_FLIPSUMMONING,
    MSG_HINT,
    MSG_SET,
    MSG_SPSUMMONING,
    MSG_SUMMONING,
    phase_to_index,
)
from yugioh_core.encoding import (
    EVENT_ENTRY_FEATURES,
    MAX_EVENT_HISTORY,
    encode_event_entry,
)

_DECLARATION_HINTS = frozenset({HINT_CODE, HINT_NUMBER, HINT_RACE, HINT_ATTRIB})

# Summon-like events store the RAW engine msg_type; entries are card+loc only.
_SUMMON_MSGS = frozenset({MSG_SUMMONING, MSG_SPSUMMONING, MSG_FLIPSUMMONING, MSG_SET})


class _Entry:
    """A stored event: its fully-encoded row plus the raw player fields.

    The row is encoded once at append time with RAW controller/turn_player in
    bytes [1]/[2]; ``to_tensor`` copies the row and overwrites just those two
    bytes with the agent-relative values (only they depend on ``agent_player``).
    """

    __slots__ = ("row", "raw_controller", "raw_turn_player")

    def __init__(self, row: np.ndarray, raw_controller: int, raw_turn_player: int) -> None:
        self.row = row
        self.raw_controller = raw_controller
        self.raw_turn_player = raw_turn_player


class EventHistoryBuffer:
    def __init__(self) -> None:
        self._buf: deque[_Entry] = deque(maxlen=MAX_EVENT_HISTORY)

    def reset(self) -> None:
        self._buf.clear()

    def append_from_enriched(
        self, enriched: list[dict], turn_count: int, current_player: int, phase: int
    ) -> None:
        for msg in enriched:
            kw = self._to_entry(msg)
            if kw is None:
                continue
            raw_controller = kw.get("controller", 0)
            kw["turn_player"] = current_player  # raw
            kw["turn_count"] = turn_count
            kw["phase"] = phase_to_index(phase)
            row = encode_event_entry(**kw)
            self._buf.append(_Entry(row, raw_controller, current_player))

    @staticmethod
    def _to_entry(msg: dict) -> dict | None:
        mt = msg.get("msg_type")
        if mt in _SUMMON_MSGS:
            return {
                "msg_type": mt,  # RAW engine msg_type
                "card_code": msg.get("code", 0),
                "controller": msg.get("controller", 0),
                "location": msg.get("location", 0),
                "sequence": msg.get("sequence", 0),
            }
        if mt == MSG_CHAINING:
            return {
                "msg_type": MSG_CHAINING,
                "card_code": msg.get("code", 0),
                "controller": msg.get("controller", 0),
                "location": msg.get("location", 0),
                "sequence": msg.get("sequence", 0),
                "desc": msg.get("desc", 0),
            }
        if mt == MSG_ATTACK:
            return {
                "msg_type": MSG_ATTACK,
                "card_code": msg.get("attacker_code", 0),
                "controller": msg.get("attacker_controller", 0),
                "location": msg.get("attacker_location", 0),
                "sequence": msg.get("attacker_sequence", 0),
                "target_code": msg.get("target_code", 0),
                "target_location": msg.get("target_location", 0),
                "target_sequence": msg.get("target_sequence", 0),
            }
        if mt == MSG_HINT and msg.get("hint_type") in _DECLARATION_HINTS:
            ht = msg["hint_type"]
            data = msg.get("data", 0)
            entry = {
                "msg_type": MSG_HINT,
                "hint_type": ht,
                "controller": msg.get("player", 0),
            }
            if ht == HINT_CODE:
                entry["card_code"] = data
            else:
                entry["hint_value"] = data
            return entry
        return None

    def to_tensor(self, agent_player: int) -> np.ndarray:
        out = np.zeros((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
        n = len(self._buf)
        for i, entry in enumerate(self._buf):
            row = MAX_EVENT_HISTORY - n + i  # right-aligned; newest at last row
            out[row] = entry.row  # cached, raw controller/turn_player in [1]/[2]
            out[row, 1] = 0 if entry.raw_controller == agent_player else 1
            out[row, 2] = 0 if entry.raw_turn_player == agent_player else 1
        return out
