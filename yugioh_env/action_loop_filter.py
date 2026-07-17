"""Action loop detection and suppression for YuGiOhEnvironment.

Detects controlled loops — a player repeatedly selecting the same action
with no net change to the game state — and suppresses the looping action so
the duel can progress.  ``ActionLoopFilter`` documents the mechanism.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
)

if TYPE_CHECKING:
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

# ─── Module-level constants ───────────────────────────────────────────────────

# Default ``sampling_start``: selection (1-based, per action key) at which
# fingerprinting begins; earlier selections are exempt.
SAMPLING_START = 2

# Max fingerprints retained per action key for recurrence detection.  Bounds
# the longest detectable loop period; real controlled loops are tiny (the known
# case is period-2), so a small window suffices.  Stored as hashes (ints).
_FP_HISTORY_MAX = 8

# Type alias: (code, controller, location, sequence, action_type, desc)
ActionKey = tuple[int, int, int, int, int, int]

# Status bits relevant to game-state comparison.
# The full status word has ~25 transient flags (CHAINING, SUMMONING, etc.)
# that flicker between snapshots; masking to these avoids false negatives.
_STATUS_MASK = 0x04000001  # STATUS_DISABLED | STATUS_FORBIDDEN
_ZONES: tuple[int, ...] = (
    LOCATION_DECK,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
)


# ─── ActionLoopFilter ────────────────────────────────────────────────────────


class ActionLoopFilter:
    """Detect and suppress controlled loops with no net change.

    A controlled loop occurs when a player voluntarily repeats the same
    action (e.g. activating a disabled card effect) and each repetition
    leaves the game state unchanged.

    Each action key is tracked independently.  Fingerprint sampling begins
    at the ``sampling_start``-th selection of a key (earlier selections are
    exempt — a one-off action cannot loop, so it pays no cost); each sampled
    selection fingerprints the game state.  Detection is by **recurrence**:

    - If the current state matches an earlier fingerprint for this key,
      the loop has returned to a state it already visited (no net
      progress) → suppress the action.
    - If the state is new → real progress → lift any suppression.

    Because it triggers on *any* recurrence (not just equality with the
    previous state), this catches period-N loops, not only period-1 —
    including loops whose only per-cycle delta is a derived status bit that
    oscillates.

    Usage (called by YuGiOhEnvironment)::

        # After each action is selected:
        self._loop_filter.record_selection(action_dict)

        # Before presenting candidates:
        if self._loop_filter.is_looping(action_dict):
            # remove from action list
    """

    def __init__(
        self,
        env: YuGiOhEnvironment,
        sampling_start: int = SAMPLING_START,
    ) -> None:
        if sampling_start < 1:
            raise ValueError(f"sampling_start must be >= 1, got {sampling_start}")
        self._env = env
        self._sampling_start = sampling_start

        self._seen: dict[ActionKey, int] = {}
        self._fp_history: dict[ActionKey, deque] = {}
        self._looping_keys: set[ActionKey] = set()

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def has_looping_actions(self) -> bool:
        """True if any action key is currently suppressed as a loop."""
        return bool(self._looping_keys)

    @staticmethod
    def action_key(action: dict) -> ActionKey:
        """Derive a canonical key from an action dict."""
        return (
            action.get("code", 0),
            action.get("controller", 0),
            action.get("location", 0),
            action.get("sequence", 0),
            action.get("category", 0),
            action.get("desc", 0),
        )

    def record_selection(self, action: dict) -> None:
        """Record that *action* was selected, updating its loop tracking.

        Structural actions (``code == 0``, e.g. phase transitions and chain
        declines) are ignored.  For a card action, if its game state matches
        one already seen for this key the action is flagged looping; a new
        state clears the flag.  Sampling starts at ``sampling_start`` (see
        ``ActionLoopFilter``).
        """
        if action.get("code", 0) == 0:
            return

        key = self.action_key(action)
        seen = self._seen.get(key, 0) + 1
        self._seen[key] = seen

        if seen < self._sampling_start:
            return  # before sampling starts: too few selections to judge a loop

        fp_now = hash(self._game_state_fingerprint())
        hist = self._fp_history.setdefault(key, deque(maxlen=_FP_HISTORY_MAX))
        if fp_now in hist:
            self._looping_keys.add(key)  # state recurred → no net progress
        else:
            self._looping_keys.discard(key)  # new state → progress → lift
        hist.append(fp_now)

    def is_looping(self, action: dict) -> bool:
        """Return True if *action* is currently suppressed as a loop."""
        return self.action_key(action) in self._looping_keys

    def reset(self) -> None:
        """Clear all tracking state (call on env reset and new turn)."""
        self._seen.clear()
        self._fp_history.clear()
        self._looping_keys.clear()

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _game_state_fingerprint(self) -> tuple:
        """Return a hashable snapshot of the current game state.

        Queries all zones for both players via ``env._duel.query_location()``.
        Returns an empty tuple if the duel has not been created yet.
        """
        duel = self._env._duel
        if duel is None:
            return ()

        gs = duel.game_state
        cards = tuple(
            (
                player,
                loc,
                card.get("sequence", 0),
                card.get("code", 0),
                card.get("position", 0),
                card.get("status", 0) & _STATUS_MASK,
            )
            for player in (0, 1)
            for loc in _ZONES
            for card in duel.query_location(player, loc)
            if card.get("code", 0) != 0
        )

        return (tuple(gs.lp), gs.phase, cards)
