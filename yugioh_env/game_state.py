"""Track game state from info messages and engine queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_CHAIN_END,
    MSG_CHAINING,
    MSG_DAMAGE,
    MSG_DRAW,
    MSG_LPUPDATE,
    MSG_MOVE,
    MSG_NEW_PHASE,
    MSG_NEW_TURN,
    MSG_PAY_LPCOST,
    MSG_RECOVER,
    MSG_START,
    MSG_WIN,
)


@dataclass
class GameState:
    """Mutable game state updated from messages."""

    lp: list[int] = field(default_factory=lambda: [8000, 8000])
    turn_count: int = 0
    phase: int = 0
    current_player: int = 0
    is_finished: bool = False
    winner: int = -1  # -1 = no winner yet, 0/1 = player, 2 = draw

    # Zone card counts per player [player0, player1]
    hand_count: list[int] = field(default_factory=lambda: [0, 0])
    deck_count: list[int] = field(default_factory=lambda: [0, 0])
    mzone_count: list[int] = field(default_factory=lambda: [0, 0])
    szone_count: list[int] = field(default_factory=lambda: [0, 0])
    grave_count: list[int] = field(default_factory=lambda: [0, 0])
    banished_count: list[int] = field(default_factory=lambda: [0, 0])
    extra_count: list[int] = field(default_factory=lambda: [0, 0])

    # Current chain count
    chain_count: int = 0

    def update(self, msg: dict) -> None:
        """Update state from a parsed message."""
        msg_type = msg.get("msg_type")
        if msg_type is None:
            return

        if msg_type == MSG_START:
            self.lp = list(msg["lp"])
            self.deck_count = list(msg["deck_count"])
            self.extra_count = list(msg["extra_count"])
            self.hand_count = [0, 0]
            self.mzone_count = [0, 0]
            self.szone_count = [0, 0]
            self.grave_count = [0, 0]
            self.banished_count = [0, 0]

        elif msg_type == MSG_NEW_TURN:
            self.current_player = msg["player"]
            self.turn_count += 1

        elif msg_type == MSG_NEW_PHASE:
            self.phase = msg["phase"]

        elif msg_type == MSG_DAMAGE:
            p = msg["player"]
            self.lp[p] = max(0, self.lp[p] - msg["amount"])

        elif msg_type == MSG_RECOVER:
            p = msg["player"]
            self.lp[p] += msg["amount"]

        elif msg_type == MSG_LPUPDATE:
            self.lp[msg["player"]] = msg["lp"]

        elif msg_type == MSG_PAY_LPCOST:
            p = msg["player"]
            self.lp[p] = max(0, self.lp[p] - msg["amount"])

        elif msg_type == MSG_DRAW:
            p = msg["player"]
            count = len(msg["cards"])
            self.hand_count[p] += count
            self.deck_count[p] = max(0, self.deck_count[p] - count)

        elif msg_type == MSG_WIN:
            self.is_finished = True
            self.winner = msg["player"]

        elif msg_type == MSG_CHAINING:
            self.chain_count += 1

        elif msg_type == MSG_CHAIN_END:
            self.chain_count = 0

        elif msg_type == MSG_MOVE:
            self._update_zone_counts(msg)

    def _update_zone_counts(self, msg: dict) -> None:
        """Update zone counts from a MOVE message."""
        prev_loc = msg.get("prev_location", 0)
        prev_con = msg.get("prev_controller", 0)
        cur_loc = msg.get("cur_location", 0)
        cur_con = msg.get("cur_controller", 0)

        loc_to_counter = {
            LOCATION_HAND: "hand_count",
            LOCATION_DECK: "deck_count",
            LOCATION_MZONE: "mzone_count",
            LOCATION_SZONE: "szone_count",
            LOCATION_GRAVE: "grave_count",
            LOCATION_BANISHED: "banished_count",
            LOCATION_EXTRA: "extra_count",
        }

        # Decrement old location
        if prev_loc in loc_to_counter:
            attr = loc_to_counter[prev_loc]
            counts = getattr(self, attr)
            counts[prev_con] = max(0, counts[prev_con] - 1)

        # Increment new location
        if cur_loc in loc_to_counter:
            attr = loc_to_counter[cur_loc]
            counts = getattr(self, attr)
            counts[cur_con] += 1

    def reset(self) -> None:
        """Reset state for a new duel."""
        self.lp = [8000, 8000]
        self.turn_count = 0
        self.phase = 0
        self.current_player = 0
        self.is_finished = False
        self.winner = -1
        self.hand_count = [0, 0]
        self.deck_count = [0, 0]
        self.mzone_count = [0, 0]
        self.szone_count = [0, 0]
        self.grave_count = [0, 0]
        self.banished_count = [0, 0]
        self.extra_count = [0, 0]
        self.chain_count = 0
