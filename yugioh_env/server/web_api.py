"""Web API endpoints for the browser-based duel UI."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from yugioh_env.models import YuGiOhAction
from yugioh_env.server.board_state import build_board_state
from yugioh_env.server.action_describer import describe_actions, describe_prompt
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

logger = logging.getLogger(__name__)

web_router = APIRouter(prefix="/api/web")


# ─── Request / response models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: int | None = None
    opponent: str | None = None


class StepRequest(BaseModel):
    action_index: int


# ─── Helpers ────────────────────────────────────────────────────────────────

_API_PHASE_NAMES = {
    0x01: "draw",
    0x02: "standby",
    0x04: "main1",
    0x08: "battle_start",
    0x10: "battle_step",
    0x20: "damage",
    0x40: "damage_calc",
    0x80: "battle",
    0x100: "main2",
    0x200: "end",
}


def _build_response(env: YuGiOhEnvironment, event_log: list[str], done: bool, reward: float) -> dict:
    """Build the unified JSON response from current env state."""
    board = build_board_state(env)
    actions = describe_actions(env._mapper, env._card_db) if not done else []
    prompt = describe_prompt(env._mapper, env._card_db) if not done else None

    # Convert absolute controller → relative side for the client
    agent = env._agent_player
    for a in actions:
        a["side"] = "mine" if a.pop("controller") == agent else "opp"

    gs = env._duel.game_state if env._duel else None

    game_state: dict[str, Any] = {}
    if gs:
        game_state = {
            "turn": gs.turn_count,
            "phase": _API_PHASE_NAMES.get(gs.phase, "unknown"),
            "is_my_turn": gs.current_player == agent,
            "chain_count": gs.chain_count,
        }

    return {
        "board": board,
        "game_state": game_state,
        "actions": actions,
        "prompt": prompt,
        "event_log": event_log,
        "done": done,
        "reward": reward,
    }


def create_web_env(config: dict | None = None) -> YuGiOhEnvironment:
    """Create a YuGiOhEnvironment for the web UI."""
    return YuGiOhEnvironment(config)


# ─── Endpoints ──────────────────────────────────────────────────────────────

@web_router.post("/reset")
def reset_duel(body: ResetRequest, request: Request) -> dict:
    """Reset (or create) a duel and return the initial state."""
    env: YuGiOhEnvironment = request.app.state.web_env
    obs = env.reset(seed=body.seed)
    return _build_response(env, obs.event_log, obs.done, obs.reward)


@web_router.post("/step")
def step_duel(body: StepRequest, request: Request) -> dict:
    """Submit an action and return the resulting state."""
    env: YuGiOhEnvironment = request.app.state.web_env
    if env._duel is None:
        return _build_response(env, ["No active duel. Call /reset first."], True, 0.0)
    obs = env.step(YuGiOhAction(action_index=body.action_index))
    return _build_response(env, obs.event_log, obs.done, obs.reward)


@web_router.get("/state")
def get_state(request: Request) -> dict:
    """Return the current duel state without advancing."""
    env: YuGiOhEnvironment = request.app.state.web_env
    if env._duel is None:
        return _build_response(env, [], True, 0.0)

    # Build response from current state (no step/reset)
    done = env._duel.game_state.is_finished if env._duel else True
    reward = 0.0
    if done and env._duel:
        winner = env._duel.game_state.winner
        if winner == env._agent_player:
            reward = 1.0
        elif winner == 1 - env._agent_player:
            reward = -1.0
    return _build_response(env, [], done, reward)
