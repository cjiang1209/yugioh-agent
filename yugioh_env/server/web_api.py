"""Web API endpoints for the browser-based duel UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from yugioh_core.string_resolver import CardTextResolver, load_sys_strings
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.event_logger import EventDescriber
from yugioh_env.models import YuGiOhAction
from yugioh_env.server.board_state import render_board
from yugioh_env.server.card_info import CardInfo, build_card_info
from yugioh_env.server.recommender import Recommendation, recommend
from yugioh_env.server.yugioh_environment import API_PHASE_NAMES, agent_reward

web_router = APIRouter(prefix="/api/web")

logger = logging.getLogger(__name__)


# ─── Request / response models ─────────────────────────────────────────────


class ResetRequest(BaseModel):
    seed: int | None = None
    opponent: str | None = None
    deck0: dict | None = None  # {"main": [int, ...], "extra": [int, ...]}
    deck1: dict | None = None
    open_cards: bool = False
    agent_player: int | str | None = None  # 0, 1, or "random"; None uses env config default
    puzzle: dict | None = None
    recommend: bool = False  # enable AI-assist action recommendation for this duel


class StepRequest(BaseModel):
    action_index: int


# ─── Helpers ────────────────────────────────────────────────────────────────


def _resolve_pending_chain(raw: list[dict], text: CardTextResolver) -> list[dict]:
    """Resolve raw chain entries to display dicts with card names and effect text."""
    return [
        {
            "chain_link": e["chain_link"],
            "card_code": e["code"],
            "card_name": text.card_name(e["code"]),
            "effect_text": text.effect_text(e["desc"]),
            "controller": e["controller"],
        }
        for e in raw
    ]


def _resolve_game_state(game_state: dict, text: CardTextResolver) -> dict:
    """Copy a raw game_state dict with phase named (raw ygopro-core int, or None
    when no duel) and pending_chain resolved to display text."""
    return {
        **game_state,
        "phase": API_PHASE_NAMES.get(game_state["phase"], "unknown"),
        "pending_chain": _resolve_pending_chain(game_state["pending_chain"], text),
    }


def _build_response(
    serving,
    action_describer: ActionDescriber,
    event_describer: EventDescriber,
    card_text_resolver: CardTextResolver,
    obs,  # YuGiOhObservation
    done: bool,
    reward: float,
    *,
    raw_frames: list[dict],
    open_cards: bool,
    recommendation: Recommendation | None = None,
) -> dict:
    """Build the unified JSON response from ServingEnv state (the ONLY builder).

    Args:
        raw_frames: Raw per-chunk snapshots from `serving.reset()`/`serving.step()`.
            GET /state passes `[]` (never advances the duel, so there is nothing
            fresh to render).
        open_cards: Per-duel display policy (set at /reset, persisted in app.state);
            when True the opponent's hidden cards are revealed at render time.
        recommendation: The recommender's output for this prompt, or None when
            AI-assist is off / unavailable / this is the read-only /state
            endpoint.
    """
    card_db = serving.card_db
    frames = [
        {
            "events": formatted,
            "board": render_board(f["board"], card_db, open_cards=open_cards),
            "game_state": _resolve_game_state(f["game_state"], card_text_resolver),
        }
        for f in raw_frames
        # Drop frames whose messages rendered to no strings (describe() can
        # return [] when a chunk holds only messages it does not materialize).
        if (formatted := event_describer.describe(f["events"], serving.agent_player))
    ]

    # Reuse the last frame's board if available (avoids redundant FFI call)
    board = (
        frames[-1]["board"]
        if frames
        else render_board(serving.capture_board(), card_db, open_cards=open_cards)
    )

    if obs is None or done:
        actions = []
        prompt = None
    else:
        actions = [d.to_dict() for d in action_describer.describe_all(obs)]
        prompt = action_describer.describe_prompt(obs)

    top_gs = _resolve_game_state(serving.capture_game_state(), card_text_resolver)
    return {
        "board": board,
        "game_state": top_gs,
        "actions": actions,
        "prompt": prompt,
        "done": done,
        "reward": reward,
        "frames": frames,
        # One object, so the index and the readouts it was computed with cannot
        # arrive out of step.
        "recommendation": (recommendation.to_dict() if recommendation is not None else None),
    }


def _resolve_recommendation(request: Request, obs) -> Recommendation | None:
    """Best-effort recommendation for the current prompt, or None.

    Returns None when AI-assist is disabled for this duel, no recommender is
    configured, the observation is terminal/empty, or inference fails. Called
    only from the mutating endpoints (/reset, /step); the read-only /state
    endpoint never recommends.
    """
    recommender = getattr(request.app.state, "recommender", None)
    if not getattr(request.app.state, "recommend_enabled", False) or recommender is None:
        return None
    if obs is None or obs.done or not obs.action_descriptors:
        return None
    try:
        return recommend(recommender, obs)
    except Exception:
        logger.warning("Recommendation failed; returning no recommendation", exc_info=True)
        return None


def create_web_env(config: dict | None = None):
    """Create a ServingEnv (core env + serving-session state) for the web UI."""
    from yugioh_env.server.serving_env import ServingEnv

    return ServingEnv(config)


def create_action_describer(env, strings_path: str | Path | None = None) -> ActionDescriber:
    """Construct the ActionDescriber the web UI shares across all requests.

    Reuses the env's `cards.cdb` connection (works for a `ServingEnv` or the
    core `YuGiOhEnvironment` directly — both expose `.card_db`). Sysstring
    labels are loaded via `load_sys_strings` (see it for path resolution);
    missing strings.conf falls back to placeholder labels.
    """
    sys_strings = load_sys_strings(strings_path)
    return ActionDescriber(env.card_db, sys_strings=sys_strings)


def create_event_describer(env, strings_path: str | Path | None = None) -> EventDescriber:
    """Construct the EventDescriber the web UI uses to format frame events.

    Mirrors `create_action_describer`: reuses the env's `cards.cdb` and loads sysstring
    labels via `load_sys_strings` (missing strings.conf falls back to
    placeholders).
    """
    sys_strings = load_sys_strings(strings_path)
    return EventDescriber(env.card_db, sys_strings=sys_strings)


def create_card_text_resolver(env, strings_path: str | Path | None = None) -> CardTextResolver:
    """Construct the CardTextResolver the web UI uses to resolve chain entries.

    Reuses the env's `cards.cdb` connection. Sysstring labels are loaded via
    `load_sys_strings`; missing strings.conf falls back to None (no sysstring
    resolution).
    """
    sys_strings = load_sys_strings(strings_path)
    return CardTextResolver(env.card_db, sys_strings=sys_strings)


# ─── Endpoints ──────────────────────────────────────────────────────────────


@web_router.post("/reset")
def reset_duel(body: ResetRequest, request: Request) -> dict:
    """Reset (or create) a duel and return the initial state."""
    serving = request.app.state.web_env
    action_describer: ActionDescriber = request.app.state.action_describer
    event_describer: EventDescriber = request.app.state.event_describer
    card_text_resolver: CardTextResolver = request.app.state.card_text_resolver
    # Display policy lives in the HTTP layer (beside recommend_enabled), not in
    # ServingEnv. Set before delegating so it reflects the request even on a 422.
    request.app.state.open_cards = body.open_cards
    try:
        obs, raw_frames = serving.reset(
            seed=body.seed,
            deck0=body.deck0,
            deck1=body.deck1,
            agent_player=body.agent_player,
            puzzle=body.puzzle,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    # Arm the recommender only after the duel resets successfully, so a failed
    # reset (422) never leaves recommend_enabled set against a stale duel.
    recommender = getattr(request.app.state, "recommender", None)
    request.app.state.recommend_enabled = bool(body.recommend and recommender is not None)
    if request.app.state.recommend_enabled:
        try:
            recommender.reseed(body.seed or 0)
        except Exception:
            logger.warning(
                "Recommender reseed failed; disabling recommendation for this duel",
                exc_info=True,
            )
            request.app.state.recommend_enabled = False
    return _build_response(
        serving,
        action_describer,
        event_describer,
        card_text_resolver,
        obs,
        obs.done,
        obs.reward,
        raw_frames=raw_frames,
        open_cards=body.open_cards,
        recommendation=_resolve_recommendation(request, obs),
    )


@web_router.get("/config")
def get_config(request: Request) -> dict:
    """Return UI capability flags (whether AI-assist recommendation is available)."""
    recommender = getattr(request.app.state, "recommender", None)
    return {"recommend_available": recommender is not None}


@web_router.get("/card/{code}")
def get_card_info(code: int, request: Request) -> CardInfo:
    """Return the printed card face for a passcode, read from cards.cdb.

    Duel-independent: answers before any /reset and never touches engine state.
    """
    info = build_card_info(code, request.app.state.web_env.card_db)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown card code: {code}")
    return info


@web_router.get("/decks")
def list_decks(request: Request) -> list[dict]:
    """List available .ydk decks from assets/decks/ with card names."""
    decks_dir = Path(__file__).resolve().parent.parent.parent / "assets" / "decks"
    card_db = request.app.state.web_env.card_db

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
    serving = request.app.state.web_env
    action_describer: ActionDescriber = request.app.state.action_describer
    event_describer: EventDescriber = request.app.state.event_describer
    card_text_resolver: CardTextResolver = request.app.state.card_text_resolver
    obs, raw_frames = serving.step(YuGiOhAction(action_index=body.action_index))
    return _build_response(
        serving,
        action_describer,
        event_describer,
        card_text_resolver,
        obs,
        obs.done,
        obs.reward,
        raw_frames=raw_frames,
        open_cards=getattr(request.app.state, "open_cards", False),
        recommendation=_resolve_recommendation(request, obs),
    )


@web_router.get("/state")
def get_state(request: Request) -> dict:
    """Return the current duel state without advancing.

    Pure read-only: never builds a fresh observation (which would mutate
    state via _make_terminal_observation or trigger FFI queries via
    _make_observation). Both `actions` and `prompt` short-circuit to
    empty when obs=None. Never returns frames (nothing new was captured).
    """
    serving = request.app.state.web_env
    action_describer: ActionDescriber = request.app.state.action_describer
    event_describer: EventDescriber = request.app.state.event_describer
    card_text_resolver: CardTextResolver = request.app.state.card_text_resolver
    done = serving.is_finished
    reward = agent_reward(serving.winner, serving.agent_player) if done else 0.0
    return _build_response(
        serving,
        action_describer,
        event_describer,
        card_text_resolver,
        None,
        done,
        reward,
        raw_frames=[],
        open_cards=getattr(request.app.state, "open_cards", False),
    )
