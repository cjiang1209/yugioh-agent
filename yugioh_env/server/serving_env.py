"""Serving-session adapter: per-chunk raw frame capture + current-state capture.

Owns the frame-capture lifecycle and raw (unhidden) capture only. Owns NO
rendering and NO display policy — describers, card_db name resolution, and the
open_cards hiding decision all live in web_api.
"""

from __future__ import annotations

from typing import Any

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_CHAINING,
)
from yugioh_env.action_space import _relativize_controller
from yugioh_env.frame_observer import DuelView


def _chain_entry(chain_link, code, desc, controller, agent_player) -> dict:
    return {
        "chain_link": chain_link,
        "code": code,
        "desc": desc,
        "controller": _relativize_controller(controller, agent_player),
    }


def _raw_pending_chain(events, agent_player) -> list[dict]:
    return [
        _chain_entry(
            m["chain_link"],
            m.get("code", 0),
            m.get("desc", 0),
            m.get("controller", 0),
            agent_player,
        )
        for m in events
        if m.get("msg_type") == MSG_CHAINING
    ]


def _capture_side(view: DuelView, player: int) -> dict:
    q, gs = view.query_location, view.game_state
    return {
        "hand": q(player, LOCATION_HAND),
        "monsters": q(player, LOCATION_MZONE),
        "spells_traps": q(player, LOCATION_SZONE),
        "grave": q(player, LOCATION_GRAVE),
        "banished": q(player, LOCATION_BANISHED),
        "extra": q(player, LOCATION_EXTRA),
        "lp": gs.lp[player],
        "deck_count": gs.deck_count[player],
        "hand_count": gs.hand_count[player],
        "extra_count": gs.extra_count[player],
    }


def capture_board(view: DuelView) -> dict:
    """Full both-sides raw board (UNHIDDEN incl. opp hand/extra). No card_db."""
    gs = view.game_state
    if gs is None:
        return {"agent": {}, "opponent": {}}
    agent = view.agent_player
    return {
        "agent": _capture_side(view, agent),
        "opponent": _capture_side(view, 1 - agent),
        "agent_player": agent,
    }


def capture_game_state(view: DuelView, events=None) -> dict:
    """Lightweight game_state — NO zone FFI. phase stays raw int|None."""
    gs = view.game_state
    if gs is None:
        return {
            "turn": 0,
            "phase": None,
            "is_my_turn": False,
            "chain_count": 0,
            "pending_chain": [],
        }
    if events is not None:
        pending = _raw_pending_chain(events, view.agent_player)
    else:
        pending = [
            _chain_entry(link.chain_link, link.code, link.desc, link.controller, view.agent_player)
            for link in gs.pending_chain
        ]
    return {
        "turn": gs.turn_count,
        "phase": gs.phase,
        "is_my_turn": gs.current_player == view.agent_player,
        "chain_count": gs.chain_count,
        "pending_chain": pending,
    }


def capture_raw_snapshot(view: DuelView, events=None) -> dict:
    return {
        "events": events if events is not None else [],
        "board": capture_board(view),
        "game_state": capture_game_state(view, events),
    }


class FrameCollector:
    """Accumulates one RawSnapshot per engine chunk. Implements FrameObserver."""

    def __init__(self) -> None:
        self._frames: list[dict] = []

    def on_chunk(self, events, view) -> None:
        self._frames.append(capture_raw_snapshot(view, events))

    def begin(self) -> None:
        """Clear the buffer at the start of a step/reset cycle."""
        self._frames = []

    def take(self) -> list[dict]:
        f = self._frames
        self._frames = []
        return f


class ServingEnv:
    """Per-duel serving-session adapter around YuGiOhEnvironment. No rendering."""

    def __init__(self, config: dict | None = None) -> None:
        # Local import (matches TrainingEnv/EvalEnv): keeps the heavy engine module
        # out of import time and lets tests patch the source module.
        from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

        self._env = YuGiOhEnvironment(config)
        self._collector = FrameCollector()
        self._env.set_frame_observer(self._collector)

    @property
    def env(self):
        return self._env

    def reset(self, **kwargs) -> tuple[Any, list[dict]]:
        self._collector.begin()
        obs = self._env.reset(**kwargs)
        return obs, self._collector.take()

    def step(self, action, **kwargs) -> tuple[Any, list[dict]]:
        self._collector.begin()
        obs = self._env.step(action, **kwargs)
        return obs, self._collector.take()

    def capture_board(self) -> dict:
        return capture_board(self._env)

    def capture_game_state(self) -> dict:
        return capture_game_state(self._env)

    def close(self) -> None:
        self._env.close()

    @property
    def current_msg(self):
        return self._env.current_msg

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def card_db(self):
        return self._env.card_db

    @property
    def is_finished(self) -> bool:
        return self._env.is_finished

    @property
    def winner(self):
        return self._env.winner

    @property
    def agent_player(self) -> int:
        return self._env.agent_player
