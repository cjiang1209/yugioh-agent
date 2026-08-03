"""HTTP bridge opponent for ygo-agent inference server.

Sends game state as JSON to ygo-agent's FastAPI server and maps
the predicted action back to this repo's action index.
"""

from __future__ import annotations

import contextlib
import logging

import requests

from yugioh_core.constants import (
    MSG_ANNOUNCE_CARD,
    MSG_ANNOUNCE_RACE,
    MSG_SELECT_COUNTER,
    MSG_SORT_CARD,
    MSG_SORT_CHAIN,
)
from yugioh_env.models import YuGiOhObservation
from yugioh_env.opponent import Opponent
from yugioh_env.ygo_agent.bridge import build_predict_input, match_response

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:3000"

# Message types the ygo-agent server can't handle: its C++ env resolves these
# internally and never presents them to the model, so the JSON API has no branch
# for them. We skip the server and pick action 0 (default/first option).
_SERVER_UNSUPPORTED_MSGS = frozenset(
    {
        MSG_SORT_CARD,
        MSG_SORT_CHAIN,
        MSG_SELECT_COUNTER,
        MSG_ANNOUNCE_RACE,
        MSG_ANNOUNCE_CARD,
    }
)


class YGOAgentOpponent(Opponent):
    """Opponent backed by a ygo-agent inference server.

    Usage::

        make_opponent("ygo-agent")                          # default localhost:3000
        make_opponent("ygo-agent:http://192.168.1.5:3000")  # custom endpoint
    """

    def __init__(self, base_url: str = DEFAULT_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._duel_id: str | None = None
        self._index: int = 0
        self._prev_action_idx: int = 0
        self._obs: YuGiOhObservation | None = None

    @property
    def needs_observation(self) -> bool:
        return True

    def set_observation(self, obs: YuGiOhObservation) -> None:
        self._obs = obs

    def _delete_session(self) -> None:
        """Best-effort delete of the current duel session."""
        if self._duel_id is not None:
            with contextlib.suppress(requests.RequestException):
                requests.delete(f"{self._base_url}/v0/duels/{self._duel_id}")
            self._duel_id = None

    def _create_session(self) -> None:
        """Create a new duel session on the server."""
        resp = requests.post(f"{self._base_url}/v0/duels")
        resp.raise_for_status()
        data = resp.json()
        self._duel_id = data["duelId"]
        self._index = data["index"]
        self._prev_action_idx = 0

    def reseed(self, seed: int) -> None:
        """Start a new duel session. Deletes the old one if any."""
        self._delete_session()
        self._create_session()
        self._obs = None

    def _reset_session(self) -> None:
        """Create a fresh duel session, discarding the current one."""
        self._delete_session()
        try:
            self._create_session()
        except requests.RequestException:
            self._duel_id = None
            self._index = 0
            self._prev_action_idx = 0

    def select_action(self, msg: dict, num_actions: int) -> int:
        if self._duel_id is None or self._obs is None:
            logger.warning("YGOAgentOpponent: no duel session or observation, returning 0")
            return 0

        # ygo-agent's C++ env handles these message types internally and
        # never presents them to the model.  We skip the server and pick
        # action 0 (default/first option).
        msg_type = msg.get("msg_type", 0)
        if msg_type in _SERVER_UNSUPPORTED_MSGS:
            return 0

        body = build_predict_input(self._obs.as_arrays(), msg, self._prev_action_idx, self._index)

        try:
            resp = requests.post(
                f"{self._base_url}/v0/duels/{self._duel_id}/predict",
                json=body,
            )
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"ygo-agent server at {self._base_url} is not reachable: {e}"
            ) from e
        except requests.RequestException as e:
            logger.warning("ygo-agent predict request failed: %s", e)
            return 0

        if resp.status_code >= 500:
            logger.warning(
                "ygo-agent server error (HTTP %d) for msg_type=%s",
                resp.status_code,
                msg.get("msg_type"),
            )
            self._reset_session()
            return 0
        if resp.status_code >= 400:
            logger.warning("ygo-agent predict HTTP %d: %s", resp.status_code, resp.text[:200])
            return 0

        data = resp.json()

        if "error" in data:
            logger.warning("ygo-agent predict error: %s", data["error"])
            # The server's PredictState may be corrupt after an error
            # (e.g. unsupported desc in select_option). Create a fresh
            # session so subsequent calls don't hit index-mismatch 500s.
            self._reset_session()
            return 0

        self._index = data["index"]
        preds = data["predict_results"]["action_preds"]
        if not preds:
            return 0

        # Pick the highest-probability action
        best = max(preds, key=lambda p: p["prob"])
        self._prev_action_idx = preds.index(best)

        # Match the server's response to our action index
        from yugioh_env.action_space import ActionMapper

        mapper = ActionMapper()
        mapper.update({**msg, "_agent_player": msg.get("player", 0)})

        action_idx = match_response(msg.get("msg_type", 0), mapper.actions, best["response"])
        return min(action_idx, num_actions - 1)
