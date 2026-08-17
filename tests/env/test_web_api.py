"""Integration tests for the web API endpoints.

Uses starlette.testclient for in-process testing (no running server).
Skips when prerequisites (lib, cards.cdb, CardScripts) are missing.
"""

import pytest


@pytest.fixture
def web_client(lib, db_path, script_dirs, deck_path):
    """Create a TestClient wrapping the FastAPI app with a configured web env."""
    # Build a standalone app with only the web router
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from yugioh_env.server.web_api import (
        create_action_describer,
        create_card_text_resolver,
        create_event_describer,
        create_web_env,
        web_router,
    )

    app = FastAPI()
    app.state.web_env = create_web_env(
        {
            "db_path": str(db_path),
            "script_dirs": [str(d) for d in script_dirs],
            "deck_path": str(deck_path),
            "opponent": "random",
            "opponent_seed": 42,
        }
    )
    app.state.action_describer = create_action_describer(app.state.web_env)
    app.state.event_describer = create_event_describer(app.state.web_env)
    app.state.card_text_resolver = create_card_text_resolver(app.state.web_env)
    app.include_router(web_router)
    return TestClient(app)


def _reset(client, seed=42, *, puzzle=None, open_cards=False):
    body = {"seed": seed, "open_cards": open_cards}
    if puzzle is not None:
        body["puzzle"] = puzzle
    resp = client.post("/api/web/reset", json=body)
    assert resp.status_code == 200
    return resp.json()


def _step(client, action_index=0):
    resp = client.post("/api/web/step", json={"action_index": action_index})
    assert resp.status_code == 200
    return resp.json()


# ─── Response shape tests ──────────────────────────────────────────────────


def test_reset_returns_valid_board(web_client):
    """Reset returns board with correct top-level keys and types."""
    data = _reset(web_client)

    assert "board" in data
    assert "player" in data["board"]
    assert "opponent" in data["board"]

    player = data["board"]["player"]
    assert isinstance(player["hand"], list)
    assert isinstance(player["monsters"], list)
    assert isinstance(player["spells_traps"], list)
    assert isinstance(player["lp"], int)
    assert isinstance(player["deck_count"], int)

    opp = data["board"]["opponent"]
    assert isinstance(opp["hand_count"], int)
    assert isinstance(opp["monsters"], list)
    assert isinstance(opp["lp"], int)

    assert isinstance(data["actions"], list)
    assert len(data["actions"]) > 0
    assert data["done"] is False
    assert data["reward"] == 0.0


def test_reset_cards_have_names(web_client):
    """Every card in hand should have a non-empty, non-Unknown name."""
    data = _reset(web_client)
    hand = data["board"]["player"]["hand"]
    assert len(hand) > 0

    for card in hand:
        assert "code" in card
        assert "name" in card
        assert "type" in card
        assert card["code"] > 0
        assert card["name"] not in ("", "Unknown", "Unknown(0)")
        assert card["type"] in ("monster", "spell", "trap")


def test_reset_extra_deck_visible_for_player(web_client):
    """Player extra_deck should be a list of card dicts; opponent keeps extra_deck_count."""
    data = _reset(web_client)
    player = data["board"]["player"]
    opp = data["board"]["opponent"]

    # Player gets full extra deck card info
    assert "extra_deck" in player
    assert isinstance(player["extra_deck"], list)
    for card in player["extra_deck"]:
        assert card["code"] > 0
        assert card["name"] not in ("", "Unknown", "Unknown(0)")
        assert card["type"] in ("monster", "spell", "trap")

    # Opponent only gets a count (no card data)
    assert "extra_deck_count" in opp
    assert isinstance(opp["extra_deck_count"], int)
    assert "extra_deck" not in opp


def test_step_returns_valid_response(web_client):
    """Step with action 0 returns the same response shape."""
    _reset(web_client)
    data = _step(web_client, action_index=0)

    assert "board" in data
    assert "game_state" in data
    assert "actions" in data
    assert "done" in data
    assert "reward" in data


def test_state_matches_after_reset(web_client):
    """GET /state should return the same board as the most recent reset."""
    reset_data = _reset(web_client)

    resp = web_client.get("/api/web/state")
    assert resp.status_code == 200
    state_data = resp.json()

    # Board should match
    assert state_data["board"]["player"]["lp"] == reset_data["board"]["player"]["lp"]
    assert state_data["board"]["opponent"]["lp"] == reset_data["board"]["opponent"]["lp"]
    assert len(state_data["board"]["player"]["hand"]) == len(reset_data["board"]["player"]["hand"])
    assert state_data["done"] == reset_data["done"]


def test_state_does_not_return_stale_frames(web_client):
    """GET /state should return empty frames (it doesn't advance the duel)."""
    _reset(web_client)

    resp = web_client.get("/api/web/state")
    assert resp.status_code == 200
    state_data = resp.json()

    assert state_data["frames"] == []


def test_state_board_present_without_frames(web_client):
    _reset(web_client, seed=7)
    state = web_client.get("/api/web/state").json()
    assert state["frames"] == []
    assert state["board"]["player"]  # from capture_board fallback


def test_action_descriptions_nonempty(web_client):
    """Every action should have a non-empty description string."""
    data = _reset(web_client)

    for action in data["actions"]:
        assert "index" in action
        assert "description" in action
        assert "category" in action
        assert isinstance(action["description"], str)
        assert len(action["description"]) > 0
        assert isinstance(action["category"], str)
        assert len(action["category"]) > 0


def test_game_state_fields(web_client):
    """Game state should have turn, phase, is_my_turn, chain_count."""
    data = _reset(web_client)
    gs = data["game_state"]

    assert "turn" in gs
    assert "phase" in gs
    assert "is_my_turn" in gs
    assert "chain_count" in gs
    assert gs["turn"] >= 1
    assert gs["phase"] in (
        "draw",
        "standby",
        "main1",
        "battle_start",
        "battle",
        "battle_step",
        "damage",
        "damage_calc",
        "main2",
        "end",
    )
    assert isinstance(gs["is_my_turn"], bool)
    assert isinstance(gs["chain_count"], int)


_FACE_DOWN_PUZZLE = {
    "player0": {"hand": [89631139], "deck": [89631139]},  # Blue-Eyes
    "player1": {
        "monster_zone": [
            {"code": 46986414, "pos": "face_down_defense", "seq": 0},  # Dark Magician
        ],
        "deck": [89631139],
    },
}


def test_opponent_face_down_hidden(web_client):
    """Face-down opponent monsters should have code=0."""
    data = _reset(web_client, puzzle=_FACE_DOWN_PUZZLE)

    opp_monsters = data["board"]["opponent"]["monsters"]
    fd = next(m for m in opp_monsters if m is not None and m.get("position") == "FACE_DOWN_DEF")
    assert fd["code"] == 0
    assert fd["name"] == "Face-down card"


def test_step_without_reset(web_client):
    """Step without reset should return done=True gracefully."""
    # Don't reset — the env has no active duel yet, but create_web_env
    # creates the env. We need to step without resetting.
    # Actually, the web_env fixture creates a fresh env with no duel.
    # reset first to have a duel, then exhaust it, then try stepping.
    data = _reset(web_client)
    # Play to completion
    for _ in range(500):
        if data["done"]:
            break
        data = _step(web_client, action_index=0)
    # Now step again on a finished duel
    data = _step(web_client, action_index=0)
    assert data["done"] is True


def test_full_duel_to_completion(web_client):
    """Play a full duel by always picking action 0 until done."""
    data = _reset(web_client)
    steps = 0
    max_steps = 500

    while not data["done"] and steps < max_steps:
        assert len(data["actions"]) > 0, f"No actions at step {steps} but game not done"
        data = _step(web_client, action_index=0)
        steps += 1

    assert data["done"] is True
    assert data["reward"] in (1.0, -1.0, 0.0)


def test_multiple_resets(web_client):
    """Resetting multiple times should work without errors."""
    for seed in [1, 2, 3]:
        data = _reset(web_client, seed=seed)
        assert data["done"] is False
        assert len(data["board"]["player"]["hand"]) > 0


def test_monster_on_field_has_stats(web_client):
    """After summoning, monster zone should show a card with attack/defense."""
    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": 89631139, "pos": "face_up_attack", "seq": 0},  # Blue-Eyes
            ],
            "deck": [89631139],
        },
        "player1": {"deck": [89631139]},
    }
    data = _reset(web_client, puzzle=puzzle)

    monsters = data["board"]["player"]["monsters"]
    bewd = next(m for m in monsters if m is not None and m.get("code", 0) > 0)
    assert bewd["name"] == "Blue-Eyes White Dragon"
    assert bewd["attack"] == 3000
    assert bewd["defense"] == 2500


def test_puzzle_reset(web_client):
    """Reset with a puzzle state should place cards correctly."""
    puzzle = {
        "player0": {
            "lp": 4000,
            "hand": [89631139],
            "monster_zone": [
                {"code": 46986414, "pos": "face_up_attack", "seq": 0},
            ],
            "deck": [89631139],
        },
        "player1": {
            "lp": 2000,
            "deck": [89631139],
        },
    }
    data = _reset(web_client, puzzle=puzzle)

    assert data["board"]["player"]["lp"] == 4000
    assert data["board"]["opponent"]["lp"] == 2000
    assert not data["done"]


# ─── Deck selection tests ─────────────────────────────────────────────────


def test_list_decks(web_client):
    """GET /decks returns available .ydk files with card names."""
    resp = web_client.get("/api/web/decks")
    assert resp.status_code == 200
    decks = resp.json()
    assert isinstance(decks, list)
    assert len(decks) >= 1

    filenames = {d["filename"] for d in decks}
    assert "blue_eyes.ydk" in filenames

    for deck in decks:
        assert "name" in deck
        assert "filename" in deck
        assert isinstance(deck["main"], list)
        assert isinstance(deck["extra"], list)
        assert 40 <= len(deck["main"]) <= 60

        # Each entry is {code, name}
        for card in deck["main"] + deck["extra"]:
            assert isinstance(card["code"], int)
            assert card["code"] > 0
            assert isinstance(card["name"], str)
            assert len(card["name"]) > 0


def test_reset_with_custom_deck(web_client):
    """Reset with explicit deck0/deck1 should succeed."""
    # Use a real deck's card IDs inline
    from yugioh_env.deck_parser import parse_ydk

    deck = parse_ydk("assets/decks/blue_eyes.ydk")
    payload = {"main": deck["main"], "extra": deck.get("extra", [])}

    resp = web_client.post(
        "/api/web/reset",
        json={
            "seed": 42,
            "deck0": payload,
            "deck1": payload,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is False
    assert len(data["board"]["player"]["hand"]) > 0


def test_reset_with_invalid_deck_returns_422(web_client):
    """Reset with a malformed deck should return 422."""
    # Missing 'main' key
    resp = web_client.post(
        "/api/web/reset",
        json={
            "seed": 42,
            "deck0": {"extra": []},
        },
    )
    assert resp.status_code == 422

    # Too few main deck cards
    resp = web_client.post(
        "/api/web/reset",
        json={
            "seed": 42,
            "deck0": {"main": [89631139] * 10, "extra": []},
        },
    )
    assert resp.status_code == 422


# ─── Frame snapshot tests ────────────────────────────────────────────────

_VALID_PHASES = {
    "draw",
    "standby",
    "main1",
    "battle_start",
    "battle_step",
    "damage",
    "damage_calc",
    "battle",
    "main2",
    "end",
    "unknown",
}


def _assert_frame_structure(frame):
    """Assert a single frame has the expected keys and types."""
    assert "events" in frame
    assert "board" in frame
    assert "game_state" in frame

    assert isinstance(frame["events"], list)
    assert len(frame["events"]) > 0
    for e in frame["events"]:
        assert isinstance(e, str)

    board = frame["board"]
    assert "player" in board
    assert "opponent" in board
    assert isinstance(board["player"]["lp"], int)
    assert isinstance(board["opponent"]["lp"], int)

    gs = frame["game_state"]
    assert isinstance(gs["turn"], int)
    assert gs["turn"] >= 1
    assert gs["phase"] in _VALID_PHASES
    assert isinstance(gs["is_my_turn"], bool)
    assert isinstance(gs["chain_count"], int)


def test_reset_returns_frames(web_client):
    """Reset should produce frames covering initial draw/phase events."""
    data = _reset(web_client)

    assert "frames" in data
    frames = data["frames"]
    assert isinstance(frames, list)
    assert len(frames) >= 1, "Reset should produce at least one frame"

    for frame in frames:
        _assert_frame_structure(frame)


def test_step_returns_frames_with_board_snapshots(web_client):
    """Step should include frames with board snapshots."""
    _reset(web_client)
    data = _step(web_client, action_index=0)

    assert "frames" in data
    frames = data["frames"]
    assert isinstance(frames, list)

    for frame in frames:
        _assert_frame_structure(frame)


def test_reset_frames_events_are_strings(web_client):
    """Frame events in the web API response must be strings (TS contract)."""
    resp = web_client.post("/api/web/reset", json={}).json()
    for frame in resp["frames"]:
        assert all(isinstance(e, str) for e in frame["events"])


def test_frames_game_state_structure(web_client):
    """Each frame's game_state should have valid fields."""
    data = _reset(web_client)

    for _ in range(10):
        if data["done"]:
            break
        data = _step(web_client, action_index=0)
        for frame in data.get("frames", []):
            gs = frame["game_state"]
            assert gs["turn"] >= 1
            assert gs["phase"] in _VALID_PHASES
            assert isinstance(gs["is_my_turn"], bool)
            assert isinstance(gs["chain_count"], int)


# ─── Open cards tests ─────────────────────────────────────────────────────


def _reset_open(client, seed=42):
    return _reset(client, seed=seed, open_cards=True)


def test_reset_open_cards_false_no_hand(web_client):
    """Default reset (open_cards=False) should not include hand/extra_deck in opponent."""
    data = _reset(web_client)
    opp = data["board"]["opponent"]
    assert "hand" not in opp
    assert "extra_deck" not in opp


def test_reset_open_cards_true_has_hand_and_extra(web_client):
    """Reset with open_cards=True should include hand and extra_deck arrays in opponent."""
    data = _reset_open(web_client)
    opp = data["board"]["opponent"]
    assert isinstance(opp["hand"], list)
    assert isinstance(opp["extra_deck"], list)
    # Standard fields still present
    assert isinstance(opp["monsters"], list)
    assert len(opp["monsters"]) == 5
    assert isinstance(opp["spells_traps"], list)
    assert len(opp["spells_traps"]) == 5


def test_open_cards_hand_has_real_codes(web_client):
    """In open_cards mode, opponent hand cards should have real codes."""
    data = _reset_open(web_client)
    opp = data["board"]["opponent"]
    assert len(opp["hand"]) > 0
    for card in opp["hand"]:
        assert card["code"] > 0, "Open opponent hand card should have a real code"
        assert card["name"] != "Face-down card"


def test_open_cards_face_down_has_real_code(web_client):
    """Face-down opponent monsters should have real codes when open_cards=True."""
    data = _reset(web_client, puzzle=_FACE_DOWN_PUZZLE, open_cards=True)

    opp_monsters = data["board"]["opponent"]["monsters"]
    fd = next(m for m in opp_monsters if m is not None and m.get("position") == "FACE_DOWN_DEF")
    assert fd["code"] > 0, "Open face-down should have real code"


def test_open_cards_frames_include_hand(web_client):
    """Frames captured during open_cards reset should include hand in opponent."""
    data = _reset_open(web_client)
    frames = data.get("frames", [])
    assert len(frames) >= 1
    for frame in frames:
        opp = frame["board"]["opponent"]
        assert isinstance(opp["hand"], list), "Frame opponent should include hand"


def test_step_preserves_open_cards(web_client):
    """After open_cards reset, subsequent steps should also include hand in opponent."""
    data = _reset_open(web_client)
    assert not data["done"]
    data = _step(web_client, action_index=0)
    assert "hand" in data["board"]["opponent"], "Step opponent should include hand"


@pytest.mark.parametrize("agent_player", [0, 1])
def test_action_controller_relativizes_per_agent_player(agent_player, web_client):
    """For both agent_player=0 and agent_player=1, every card-bearing
    action in the web API response must carry a relativized controller
    value (0 = agent's own card, 1 = opponent's card).

    Pins the web JSON layer specifically. The lower layers
    (extractor → describer) are pinned by Test C in
    tests/env/test_action_space.py and B1/B2 in tests/env/test_observation.py.

    Pre-fix bug: web_api re-derived `side` from `controller == agent_player`,
    which inverted "mine"/"opp" when agent_player=1. This test catches any
    future re-derivation regression in the web layer.
    """
    response = web_client.post(
        "/api/web/reset",
        json={"seed": 42, "agent_player": agent_player},
    )
    assert response.status_code == 200, response.text
    state = response.json()

    card_actions_seen = 0
    for _ in range(8):
        actions = state["actions"]
        for a in actions:
            if a.get("card_code", 0) == 0:
                continue
            assert "side" not in a, f"action carries legacy side field: {a!r}"
            assert "controller" in a, f"action missing controller: {a!r}"
            assert a["controller"] in (0, 1), f"controller must be 0 or 1, got {a['controller']!r}"
            card_actions_seen += 1
        if card_actions_seen >= 1 or state.get("done"):
            break
        if not actions:
            break
        # Web API returns only legal actions; step the first by its index.
        step_response = web_client.post(
            "/api/web/step",
            json={"action_index": actions[0]["index"]},
        )
        assert step_response.status_code == 200, step_response.text
        state = step_response.json()

    assert card_actions_seen >= 1, (
        "Test inconclusive: no card-bearing action observed in 8 prompts."
    )


def test_state_response_carries_populated_card(web_client):
    """After the parser dedup, the web UI's board-state path now uses
    the strict parser. Pin that real engine output produces at least
    one card on the board with a populated `code` field — would catch
    any accidental regression where the strict parser raised on
    something the lenient one tolerated.
    """
    _reset(web_client)
    resp = web_client.get("/api/web/state")
    assert resp.status_code == 200, resp.text
    state = resp.json()
    board = state.get("board", {})
    # Walk both sides; each side dict has list-valued keys (hand, monsters,
    # spells_traps, graveyard, banished, extra_deck, etc.) holding either
    # card dicts (with `code`) or None for empty zone slots.
    seen_populated_card = False
    for side_key in ("player", "opponent"):
        side = board.get(side_key, {})
        if not isinstance(side, dict):
            continue
        for value in side.values():
            if not isinstance(value, list):
                continue
            for slot in value:
                if isinstance(slot, dict) and slot.get("code"):
                    seen_populated_card = True
                    break
            if seen_populated_card:
                break
        if seen_populated_card:
            break
    assert seen_populated_card, (
        f"No populated card found on the board after reset; the strict "
        f"parser may be raising on something the lenient one tolerated. "
        f"Board snapshot: {board!r}"
    )


def test_reset_game_state_pending_chain_resolved(web_client):
    resp = web_client.post("/api/web/reset", json={}).json()
    gs = resp["game_state"]
    assert "pending_chain" in gs and isinstance(gs["pending_chain"], list)
    for e in gs["pending_chain"]:
        assert set(e) >= {"chain_link", "card_code", "card_name", "effect_text", "controller"}
    for frame in resp["frames"]:
        assert "pending_chain" in frame["game_state"]


def _make_app_with_recommender(db_path, script_dirs, deck_path, recommender):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from yugioh_env.server.web_api import (
        create_action_describer,
        create_card_text_resolver,
        create_event_describer,
        create_web_env,
        web_router,
    )

    app = FastAPI()
    app.state.web_env = create_web_env(
        {
            "db_path": str(db_path),
            "script_dirs": [str(d) for d in script_dirs],
            "deck_path": str(deck_path),
            "opponent": "random",
            "opponent_seed": 42,
        }
    )
    app.state.action_describer = create_action_describer(app.state.web_env)
    app.state.event_describer = create_event_describer(app.state.web_env)
    app.state.card_text_resolver = create_card_text_resolver(app.state.web_env)
    app.state.recommender = recommender
    app.state.recommend_enabled = False
    app.include_router(web_router)
    return TestClient(app)


class _FakeRec:
    """Non-network fake recommender: picks the last (dense) legal slot."""

    def select_action(self, obs):
        return int(obs.action_mask.sum()) - 1, None

    def reseed(self, seed):
        pass


def test_recommend_absent_when_flag_off(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _FakeRec())
    resp = client.post("/api/web/reset", json={"seed": 42})
    assert resp.status_code == 200
    assert resp.json()["recommended_action_index"] is None


def test_recommend_present_and_legal_when_flag_on(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _FakeRec())
    resp = client.post("/api/web/reset", json={"seed": 42, "recommend": True})
    assert resp.status_code == 200
    data = resp.json()
    rec_idx = data["recommended_action_index"]
    assert rec_idx is not None
    offered = {a["index"] for a in data["actions"]}
    assert rec_idx in offered


def test_recommend_none_without_recommender(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, None)
    resp = client.post("/api/web/reset", json={"seed": 42, "recommend": True})
    assert resp.status_code == 200
    assert resp.json()["recommended_action_index"] is None


def test_state_never_recommends(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _FakeRec())
    client.post("/api/web/reset", json={"seed": 42, "recommend": True})
    resp = client.get("/api/web/state")
    assert resp.status_code == 200
    assert resp.json()["recommended_action_index"] is None


def test_config_reports_recommend_available_true(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _FakeRec())
    resp = client.get("/api/web/config")
    assert resp.status_code == 200
    assert resp.json() == {"recommend_available": True}


def test_config_reports_recommend_available_false(lib, db_path, script_dirs, deck_path):
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, None)
    resp = client.get("/api/web/config")
    assert resp.status_code == 200
    assert resp.json() == {"recommend_available": False}


class _RaisingReseedRec(_FakeRec):
    """Fake recommender whose reseed() raises (e.g. offline ygo-agent server)."""

    def reseed(self, seed):
        raise RuntimeError("recommender server down")


def test_reset_survives_reseed_failure(lib, db_path, script_dirs, deck_path):
    """A recommender whose reseed() raises must not 500 the /reset. The duel
    starts normally and recommendation is silently disabled for the duel."""
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _RaisingReseedRec())
    resp = client.post("/api/web/reset", json={"seed": 42, "recommend": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is False
    assert data["recommended_action_index"] is None


class _RaisingSelectRec(_FakeRec):
    """Fake recommender whose select_action raises mid-duel."""

    def select_action(self, obs):
        raise RuntimeError("recommender server down")


def test_step_survives_recommend_failure(lib, db_path, script_dirs, deck_path):
    """A recommender that throws during inference must not 500 a /step; the
    duel continues and no recommendation is returned."""
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _RaisingSelectRec())
    reset = client.post("/api/web/reset", json={"seed": 42, "recommend": True})
    assert reset.status_code == 200
    assert reset.json()["recommended_action_index"] is None
    step = client.post("/api/web/step", json={"action_index": 0})
    assert step.status_code == 200
    assert step.json()["recommended_action_index"] is None


def test_failed_reset_does_not_arm_recommender(lib, db_path, script_dirs, deck_path):
    """A reset that 422s must not leave recommend_enabled set. Arming happens
    only after env.reset() succeeds, so a later reset without recommend stays
    off (no recommendation leaks from the failed request)."""
    client = _make_app_with_recommender(db_path, script_dirs, deck_path, _FakeRec())
    # Malformed deck -> env.reset raises ValueError -> 422.
    bad = client.post(
        "/api/web/reset",
        json={"seed": 42, "recommend": True, "deck0": {"main": [123456789]}},
    )
    assert bad.status_code == 422
    assert client.app.state.recommend_enabled is False
    # A subsequent reset that does NOT request recommendation stays off.
    ok = client.post("/api/web/reset", json={"seed": 42})
    assert ok.status_code == 200
    assert ok.json()["recommended_action_index"] is None


def test_card_info_returns_printed_face(web_client):
    """GET /card/{code} returns the printed face straight from cards.cdb.

    No /reset here, deliberately: the endpoint is duel-independent and must
    answer before a duel exists, without touching engine state. Do not add a
    reset to these card tests — it would remove that coverage.
    """
    resp = web_client.get("/api/web/card/44508094")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "Stardust Dragon"
    assert card["card_type"] == "monster"
    assert card["typeline"] == ["Dragon", "Synchro", "Effect"]
    assert card["attribute"] == "WIND"
    assert card["race"] == "Dragon"
    assert card["level"] == 8
    assert card["level_kind"] == "level"
    assert card["attack"] == 2500
    assert card["defense"] == 2000
    assert card["scales"] is None
    assert card["link_arrows"] is None
    # The text is upstream data, so assert that it arrived and that CRLF was
    # normalized away — not the wording.
    assert card["desc"]
    assert "\r" not in card["desc"]


def test_card_info_link_monster_has_arrows(web_client):
    """Double Headed Anger Knuckle — arrows present, DEF suppressed."""
    card = web_client.get("/api/web/card/146746").json()
    assert card["level_kind"] == "link"
    assert card["defense"] is None
    assert set(card["link_arrows"]) == {"RIGHT", "BOTTOM"}


def test_card_info_unknown_code_returns_404(web_client):
    resp = web_client.get("/api/web/card/99999999")
    assert resp.status_code == 404
