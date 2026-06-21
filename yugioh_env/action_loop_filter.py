"""Action loop detection and suppression for YuGiOhEnvironment.

Detects controlled loops — voluntary repetitions of the same action
that produce no net change to the game state — and suppresses the
looping action to break the cycle.

Each action key is tracked independently — intervening selections of
other actions do not reset a key's loop count.  Game-state
fingerprinting is deferred until a key has been selected
``threshold - 1`` times, so the common case (one-off actions) pays no
fingerprint cost.  A looping action repeats at most ``threshold``
times before suppression.
"""

from __future__ import annotations

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

LOOP_DETECTION_THRESHOLD = 3

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

    Each action key is tracked independently via a per-key selection
    count and game-state fingerprint.  Fingerprinting is deferred: the
    first ``threshold - 2`` selections of a key are free.

    At selection ``threshold - 1``, a "before" game-state snapshot is
    captured.  On the ``threshold``-th selection (and beyond), an
    "after" snapshot is compared with the stored "before":

    - If the game state is unchanged → the action is suppressed.
    - If the game state changed → the new snapshot becomes the next
      "before" (the action had a net change but may loop again).

    A looping action therefore repeats at most ``threshold`` times.

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
        threshold: int = LOOP_DETECTION_THRESHOLD,
    ) -> None:
        if threshold < 2:
            raise ValueError(f"threshold must be >= 2, got {threshold}")
        self._env = env
        self._threshold = threshold

        self._seen: dict[ActionKey, int] = {}
        self._pending_fp: dict[ActionKey, tuple] = {}
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
        """Record that *action* was selected.

        Structural actions (code == 0, e.g. phase transitions and chain
        declines) are ignored — only card-specific actions are tracked.

        No game-state fingerprinting until the key has been selected
        ``threshold - 1`` times.  At that point a "before" snapshot is
        captured; on the ``threshold``-th selection the "after" is
        compared.  If the game state is unchanged the action is
        suppressed — so a looping action repeats at most ``threshold``
        times.
        """
        if action.get("code", 0) == 0:
            return

        key = self.action_key(action)
        seen = self._seen.get(key, 0) + 1
        self._seen[key] = seen

        if seen < self._threshold - 1:
            return

        fp_now = self._game_state_fingerprint()

        if seen == self._threshold - 1:
            self._pending_fp[key] = fp_now
            return

        # seen >= threshold — compare "after" with stored "before"
        if fp_now == self._pending_fp[key]:
            self._looping_keys.add(key)
        else:
            self._looping_keys.discard(key)

        self._pending_fp[key] = fp_now

    def is_looping(self, action: dict) -> bool:
        """Return True if *action* is currently suppressed as a loop."""
        return self.action_key(action) in self._looping_keys

    def reset(self) -> None:
        """Clear all tracking state (call on env reset and new turn)."""
        self._seen.clear()
        self._pending_fp.clear()
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
