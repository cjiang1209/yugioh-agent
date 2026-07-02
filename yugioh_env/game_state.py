"""Track game state from info messages and engine queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_CHAIN_DISABLED,
    MSG_CHAIN_END,
    MSG_CHAIN_NEGATED,
    MSG_CHAIN_SOLVED,
    MSG_CHAIN_SOLVING,
    MSG_CHAINED,
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

logger = logging.getLogger(__name__)


class ChainStatus(StrEnum):
    """Lifecycle state of a single chain link."""

    BUILDING = "building"
    SOLVING = "solving"
    SOLVED = "solved"
    NEGATED = "negated"
    DISABLED = "disabled"


@dataclass
class ChainLink:
    """One link on the current chain.

    ``controller`` is the RAW engine controller (0/1). Relativize to
    agent/opponent at read sites (encoder, web JSON) — never store the
    relativized value, matching the GameState.current_player convention.
    """

    code: int
    desc: int
    controller: int
    location: int
    sequence: int
    chain_link: int  # 1-based
    status: ChainStatus = ChainStatus.BUILDING


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

    # Current chain, in build order. Each link carries its lifecycle status.
    # Uncapped; the RL tensor cap (MAX_PENDING_CHAIN) is applied at encode time.
    pending_chain: list[ChainLink] = field(default_factory=list)

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
            self.pending_chain.append(
                ChainLink(
                    code=msg.get("code", 0),
                    desc=msg.get("desc", 0),
                    controller=msg.get("controller", 0),
                    location=msg.get("location", 0),
                    sequence=msg.get("sequence", 0),
                    chain_link=self.chain_count,
                )
            )

        elif msg_type == MSG_CHAINED:
            # Build-phase echo; carries chain_link but changes no state.
            pass

        elif msg_type == MSG_CHAIN_SOLVING:
            self._set_link_status(msg.get("chain_link", 0), ChainStatus.SOLVING)

        elif msg_type == MSG_CHAIN_SOLVED:
            self._set_link_status(msg.get("chain_link", 0), ChainStatus.SOLVED)

        elif msg_type == MSG_CHAIN_NEGATED:
            self._set_link_status(msg.get("chain_link", 0), ChainStatus.NEGATED)

        elif msg_type == MSG_CHAIN_DISABLED:
            self._set_link_status(msg.get("chain_link", 0), ChainStatus.DISABLED)

        elif msg_type == MSG_CHAIN_END:
            self.chain_count = 0
            self.pending_chain.clear()

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

    def _set_link_status(self, chain_link: int, status: ChainStatus) -> None:
        """Stamp status on the link at index ``chain_link - 1`` (1-based).

        Links are appended in order, so ``chain_link`` maps directly to a list
        index. The entry's own ``chain_link`` is verified against the requested
        value as a consistency check.

        NEGATED/DISABLED are terminal and take precedence: a later SOLVED
        must not overwrite them. An out-of-range or mismatched chain_link is
        logged and ignored (wire-format robustness — never crash a live duel).
        """
        index = chain_link - 1
        if not 0 <= index < len(self.pending_chain):
            logger.warning(
                "Chain resolution referenced unknown chain_link %d (have %d links)",
                chain_link,
                len(self.pending_chain),
            )
            return
        link = self.pending_chain[index]
        if link.chain_link != chain_link:
            logger.warning(
                "Chain link at index %d has chain_link %d, expected %d",
                index,
                link.chain_link,
                chain_link,
            )
            return
        if link.status in (ChainStatus.NEGATED, ChainStatus.DISABLED):
            return
        link.status = status

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
        self.pending_chain.clear()
