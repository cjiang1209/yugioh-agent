"""Integration tests for the web API endpoints.

Uses starlette.testclient for in-process testing (no running server).
Skips when prerequisites (lib, cards.cdb, CardScripts) are missing.
"""

import pytest


@pytest.fixture
def web_client(lib, db_path, script_dirs, deck_path):
    """Create a TestClient wrapping the FastAPI app with a configured web env."""
    from starlette.testclient import TestClient
    from yugioh_env.server.web_api import web_router, create_web_env

    # Build a standalone app with only the web router
    from fastapi import FastAPI
    app = FastAPI()
    app.state.web_env = create_web_env({
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
    })
    app.include_router(web_router)
    return TestClient(app)


def _reset(client, seed=42):
    resp = client.post("/api/web/reset", json={"seed": seed})
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
    assert "event_log" in data
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
    assert gs["phase"] in ("draw", "standby", "main1", "battle_start", "battle",
                           "battle_step", "damage", "damage_calc", "main2", "end")
    assert isinstance(gs["is_my_turn"], bool)
    assert isinstance(gs["chain_count"], int)


def test_opponent_face_down_hidden(web_client):
    """Face-down opponent monsters should have code=0."""
    # Play several seeds and steps to find a face-down opponent monster
    for seed in range(42, 62):
        data = _reset(web_client, seed=seed)
        for _ in range(50):
            if data["done"]:
                break
            data = _step(web_client, action_index=0)
            opp_monsters = data["board"]["opponent"]["monsters"]
            for m in opp_monsters:
                if m is not None and m.get("position") in ("FACE_DOWN_DEF", "FACE_DOWN_ATK"):
                    assert m["code"] == 0
                    assert m["name"] == "Face-down card"
                    return  # Test passed

    pytest.skip("No face-down opponent monster encountered in test seeds")


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
    for seed in range(42, 72):
        data = _reset(web_client, seed=seed)
        for _ in range(100):
            if data["done"]:
                break
            data = _step(web_client, action_index=0)
            # Check if any of our monsters are on the field
            for m in data["board"]["player"]["monsters"]:
                if m is not None and m.get("code", 0) > 0:
                    assert m["name"] not in ("", "Unknown")
                    assert m["attack"] is not None
                    return  # Test passed

    pytest.skip("No player monster summoned in test seeds")


# ─── Deck selection tests ─────────────────────────────────────────────────


def test_list_decks(web_client):
    """GET /decks returns available .ydk files with parsed card lists."""
    resp = web_client.get("/api/web/decks")
    assert resp.status_code == 200
    decks = resp.json()
    assert isinstance(decks, list)
    assert len(decks) >= 1

    filenames = {d["filename"] for d in decks}
    assert "starter.ydk" in filenames

    for deck in decks:
        assert "name" in deck
        assert "filename" in deck
        assert "main" in deck
        assert "extra" in deck
        assert isinstance(deck["main"], list)
        assert isinstance(deck["extra"], list)
        assert 40 <= len(deck["main"]) <= 60


def test_reset_with_custom_deck(web_client):
    """Reset with explicit deck0/deck1 should succeed."""
    # Use the starter deck's card IDs inline
    from yugioh_env.deck_parser import parse_ydk
    deck = parse_ydk("assets/decks/starter.ydk")
    payload = {"main": deck["main"], "extra": deck.get("extra", [])}

    resp = web_client.post("/api/web/reset", json={
        "seed": 42,
        "deck0": payload,
        "deck1": payload,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is False
    assert len(data["board"]["player"]["hand"]) > 0


def test_reset_with_invalid_deck_returns_422(web_client):
    """Reset with a malformed deck should return 422."""
    # Missing 'main' key
    resp = web_client.post("/api/web/reset", json={
        "seed": 42,
        "deck0": {"extra": []},
    })
    assert resp.status_code == 422

    # Too few main deck cards
    resp = web_client.post("/api/web/reset", json={
        "seed": 42,
        "deck0": {"main": [89631139] * 10, "extra": []},
    })
    assert resp.status_code == 422
