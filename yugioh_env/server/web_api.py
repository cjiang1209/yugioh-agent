"""Web API endpoints for the browser-based duel UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from yugioh_core.string_resolver import load_sys_strings
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.models import YuGiOhAction
from yugioh_env.server.board_state import build_board_state
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

web_router = APIRouter(prefix="/api/web")


# ─── Request / response models ─────────────────────────────────────────────


class ResetRequest(BaseModel):
    seed: int | None = None
    opponent: str | None = None
    deck0: dict | None = None  # {"main": [int, ...], "extra": [int, ...]}
    deck1: dict | None = None
    open_cards: bool = False
    agent_player: int | str | None = None  # 0, 1, or "random"; None uses env config default
    puzzle: dict | None = None


class StepRequest(BaseModel):
    action_index: int


# ─── Helpers ────────────────────────────────────────────────────────────────


def _build_response(
    env: YuGiOhEnvironment,
    describer: ActionDescriber,
    obs,  # YuGiOhObservation
    done: bool,
    reward: float,
    *,
    include_frames: bool = False,
) -> dict:
    """Build the unified JSON response from current env state.

    Args:
        include_frames: When True, attach env.last_frames (only set this after
            reset/step calls that ran _process_to_agent_choice, so the frames
            are fresh).  GET /state and multi-select steps must leave this
            False to avoid returning stale frames from a prior action.
    """
    frames = env.last_frames if include_frames else []

    # Reuse the last frame's board if available (avoids redundant FFI call)
    board = frames[-1]["board"] if frames else build_board_state(env, open_cards=env._open_cards)

    if obs is None or done:
        actions = []
        prompt = None
    else:
        actions = [d.to_dict() for d in describer.describe_all(obs)]
        prompt = describer.describe_prompt(obs)

    return {
        "board": board,
        "game_state": env._build_game_state_dict(),
        "actions": actions,
        "prompt": prompt,
        "done": done,
        "reward": reward,
        "frames": frames,
    }


def create_web_env(config: dict | None = None) -> YuGiOhEnvironment:
    """Create a YuGiOhEnvironment for the web UI."""
    return YuGiOhEnvironment(config)


def create_describer(
    env: YuGiOhEnvironment, strings_path: str | Path | None = None
) -> ActionDescriber:
    """Construct the ActionDescriber the web UI shares across all requests.

    Reuses the env's `cards.cdb` connection. Sysstring labels are loaded via
    `load_sys_strings` (see it for path resolution); missing strings.conf falls
    back to placeholder labels.
    """
    sys_strings = load_sys_strings(strings_path)
    return ActionDescriber(env._card_db, sys_strings=sys_strings)


# ─── Endpoints ──────────────────────────────────────────────────────────────


@web_router.post("/reset")
def reset_duel(body: ResetRequest, request: Request) -> dict:
    """Reset (or create) a duel and return the initial state."""
    env: YuGiOhEnvironment = request.app.state.web_env
    describer: ActionDescriber = request.app.state.describer
    try:
        obs = env.reset(
            seed=body.seed,
            deck0=body.deck0,
            deck1=body.deck1,
            open_cards=body.open_cards,
            agent_player=body.agent_player,
            puzzle=body.puzzle,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _build_response(env, describer, obs, obs.done, obs.reward, include_frames=True)


@web_router.get("/decks")
def list_decks(request: Request) -> list[dict]:
    """List available .ydk decks from assets/decks/ with card names."""
    decks_dir = Path(__file__).resolve().parent.parent.parent / "assets" / "decks"
    card_db = request.app.state.web_env._card_db

    # Parse all decks first to collect codes
    parsed = []
    all_codes: set[int] = set()
    for ydk_path in sorted(decks_dir.glob("*.ydk")):
        deck = parse_ydk(ydk_path)
        main = deck["main"]
        extra = deck.get("extra", [])
        all_codes.update(main)
        all_codes.update(extra)
        parsed.append((ydk_path, main, extra))

    # Batch-fetch all card names in one query
    names = card_db.get_card_names_batch(all_codes)

    result = []
    for ydk_path, main, extra in parsed:
        name = ydk_path.stem.replace("_", " ").title()
        result.append(
            {
                "name": name,
                "filename": ydk_path.name,
                "main": [{"code": c, "name": names.get(c, f"Unknown({c})")} for c in main],
                "extra": [{"code": c, "name": names.get(c, f"Unknown({c})")} for c in extra],
            }
        )
    return result


@web_router.post("/step")
def step_duel(body: StepRequest, request: Request) -> dict:
    """Submit an action and return the resulting state."""
    env: YuGiOhEnvironment = request.app.state.web_env
    describer: ActionDescriber = request.app.state.describer
    if env._duel is None:
        return _build_response(env, describer, None, True, 0.0)
    obs = env.step(YuGiOhAction(action_index=body.action_index))
    return _build_response(env, describer, obs, obs.done, obs.reward, include_frames=True)


@web_router.get("/state")
def get_state(request: Request) -> dict:
    """Return the current duel state without advancing.

    Pure read-only: never builds a fresh observation (which would mutate
    state via _make_terminal_observation or trigger FFI queries via
    _make_observation). Both `actions` and `prompt` short-circuit to
    empty when obs=None.
    """
    env: YuGiOhEnvironment = request.app.state.web_env
    describer: ActionDescriber = request.app.state.describer
    if env._duel is None:
        return _build_response(env, describer, None, True, 0.0)

    done = env._duel.game_state.is_finished if env._duel else True
    reward = 0.0
    if done and env._duel:
        winner = env._duel.game_state.winner
        if winner == env._agent_player:
            reward = 1.0
        elif winner == 1 - env._agent_player:
            reward = -1.0
    return _build_response(env, describer, None, done, reward)
