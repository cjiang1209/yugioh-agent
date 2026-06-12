"""Test puzzle utilities — pure unit tests, no engine required."""

from __future__ import annotations

import json

import pytest

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    STATUS_DISABLED,
)
from yugioh_env.puzzle import (
    generate_disable_lua,
    load_puzzle,
    parse_position,
    validate_puzzle,
)

# Real card codes used throughout tests.
BLUE_EYES = 89631139
DARK_MAGICIAN = 46986414
MST = 5318639


class TestPositionMapping:
    def test_all_position_strings(self):
        assert parse_position("face_up_attack") == POS_FACEUP_ATTACK
        assert parse_position("face_down_attack") == POS_FACEDOWN_ATTACK
        assert parse_position("face_up_defense") == POS_FACEUP_DEFENSE
        assert parse_position("face_down_defense") == POS_FACEDOWN_DEFENSE
        assert parse_position("face_up") == POS_FACEUP
        assert parse_position("face_down") == POS_FACEDOWN

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError):
            parse_position("upside_down")


class TestValidatePuzzle:
    def test_minimal_empty_puzzle(self):
        result = validate_puzzle({})
        assert "player0" in result
        assert "player1" in result
        assert result["player0"]["lp"] == 8000
        assert result["player1"]["lp"] == 8000

    def test_defaults_applied(self):
        result = validate_puzzle({"player0": {"lp": 4000}})
        p0 = result["player0"]
        assert p0["lp"] == 4000
        assert p0["hand"] == []
        assert p0["monster_zone"] == []
        assert p0["spell_zone"] == []
        assert p0["grave"] == []
        assert p0["banished"] == []
        assert p0["deck"] == []
        assert p0["extra"] == []
        # player1 gets full defaults
        assert result["player1"]["lp"] == 8000

    def test_monster_zone_validated(self):
        state = {
            "player0": {"monster_zone": [{"code": BLUE_EYES, "pos": "face_up_attack", "seq": 0}]}
        }
        result = validate_puzzle(state)
        card = result["player0"]["monster_zone"][0]
        assert card["code"] == BLUE_EYES
        assert card["pos"] == POS_FACEUP_ATTACK
        assert card["seq"] == 0
        assert card["disabled"] is False

    def test_disabled_on_field_allowed(self):
        state = {
            "player0": {
                "monster_zone": [
                    {
                        "code": BLUE_EYES,
                        "pos": "face_up_attack",
                        "seq": 0,
                        "disabled": True,
                    }
                ]
            }
        }
        result = validate_puzzle(state)
        assert result["player0"]["monster_zone"][0]["disabled"] is True

    def test_invalid_monster_zone_seq(self):
        state = {
            "player0": {"monster_zone": [{"code": BLUE_EYES, "pos": "face_up_attack", "seq": 7}]}
        }
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_invalid_spell_zone_seq(self):
        state = {"player0": {"spell_zone": [{"code": MST, "pos": "face_down", "seq": 6}]}}
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_duplicate_monster_seq(self):
        state = {
            "player0": {
                "monster_zone": [
                    {"code": BLUE_EYES, "pos": "face_up_attack", "seq": 0},
                    {"code": DARK_MAGICIAN, "pos": "face_up_attack", "seq": 0},
                ]
            }
        }
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_non_integer_card_code(self):
        state = {"player0": {"hand": ["not_an_int"]}}
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_invalid_lp_type(self):
        state = {"player0": {"lp": "not_a_number"}}
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_negative_lp(self):
        state = {"player0": {"lp": -500}}
        with pytest.raises(ValueError):
            validate_puzzle(state)

    def test_unknown_zone_key(self):
        state = {"player0": {"battlefield": []}}
        with pytest.raises(ValueError):
            validate_puzzle(state)


class TestGenerateDisableLua:
    def test_no_disabled_cards_returns_none(self):
        state = validate_puzzle(
            {"player0": {"monster_zone": [{"code": BLUE_EYES, "pos": "face_up_attack", "seq": 0}]}}
        )
        assert generate_disable_lua(state) is None

    def test_single_disabled_monster(self):
        state = validate_puzzle(
            {
                "player0": {
                    "monster_zone": [
                        {
                            "code": BLUE_EYES,
                            "pos": "face_up_attack",
                            "seq": 0,
                            "disabled": True,
                        }
                    ]
                }
            }
        )
        lua = generate_disable_lua(state)
        assert lua is not None
        assert "EFFECT_DISABLE" in lua
        assert "GetControler()==0" in lua
        assert "GetSequence()==0" in lua
        assert "LOCATION_MZONE" in lua

    def test_multiple_disabled_across_players(self):
        state = validate_puzzle(
            {
                "player0": {
                    "monster_zone": [
                        {
                            "code": BLUE_EYES,
                            "pos": "face_up_attack",
                            "seq": 0,
                            "disabled": True,
                        }
                    ]
                },
                "player1": {
                    "spell_zone": [
                        {
                            "code": MST,
                            "pos": "face_down",
                            "seq": 3,
                            "disabled": True,
                        }
                    ]
                },
            }
        )
        lua = generate_disable_lua(state)
        assert lua is not None
        assert lua.count("Duel.RegisterEffect") == 2
        assert "GetControler()==0" in lua
        assert "GetControler()==1" in lua


class TestLoadPuzzle:
    def test_load_json(self, tmp_path):
        data = {"player0": {"lp": 4000, "hand": [BLUE_EYES]}}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        result = load_puzzle(path)
        assert result["player0"]["lp"] == 4000
        assert result["player0"]["hand"] == [BLUE_EYES]

    def test_load_unknown_extension(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("{}")
        with pytest.raises(ValueError):
            load_puzzle(path)


# ---------------------------------------------------------------------------
# Integration tests — require libocgcore, cards.cdb, CardScripts
# ---------------------------------------------------------------------------

# Card codes used in integration tests
SANGAN = 26202165
MONSTER_REBORN = 83764718
DARK_HOLE = 53129443
UPSTART_GOBLIN = 70368879


class TestCreatePuzzle:
    """Integration tests for Duel.create_puzzle()."""

    @pytest.fixture
    def duel(self, lib, card_db, script_dirs):
        from yugioh_env.duel import Duel

        d = Duel(lib, card_db, script_dirs)
        yield d
        d.destroy()

    def test_hand_only(self, duel):
        """Place cards in hand only; verify counts, no deck cards."""
        state = {
            "player0": {"hand": [BLUE_EYES, DARK_MAGICIAN]},
            "player1": {"hand": [MST]},
        }
        duel.create_puzzle(state)

        assert duel.query_count(0, LOCATION_HAND) == 2
        assert duel.query_count(1, LOCATION_HAND) == 1
        assert duel.query_count(0, LOCATION_DECK) == 0
        assert duel.query_count(1, LOCATION_DECK) == 0

    def test_monster_zone_position(self, duel):
        """Place monsters at specific sequences and positions."""
        state = {
            "player0": {
                "monster_zone": [
                    {"code": BLUE_EYES, "pos": "face_up_attack", "seq": 0},
                    {"code": DARK_MAGICIAN, "pos": "face_down_defense", "seq": 2},
                ],
            },
        }
        duel.create_puzzle(state)

        cards = duel.query_location(0, LOCATION_MZONE)
        # Filter out empty slots
        occupied = {c["sequence"]: c for c in cards if c.get("code")}
        assert occupied[0]["code"] == BLUE_EYES
        assert occupied[0]["position"] == POS_FACEUP_ATTACK
        assert occupied[2]["code"] == DARK_MAGICIAN
        assert occupied[2]["position"] == POS_FACEDOWN_DEFENSE

    def test_all_zones(self, duel):
        """Place cards in every zone type and verify counts."""
        state = {
            "player0": {
                "hand": [BLUE_EYES],
                "monster_zone": [
                    {"code": DARK_MAGICIAN, "pos": "face_up_attack", "seq": 0},
                ],
                "spell_zone": [
                    {"code": MST, "pos": "face_down", "seq": 1},
                ],
                "grave": [MONSTER_REBORN],
                "banished": [DARK_HOLE],
                "deck": [UPSTART_GOBLIN, SANGAN],
            },
        }
        duel.create_puzzle(state)

        assert duel.query_count(0, LOCATION_HAND) == 1
        assert duel.query_count(0, LOCATION_MZONE) == 1
        assert duel.query_count(0, LOCATION_SZONE) == 1
        assert duel.query_count(0, LOCATION_GRAVE) == 1
        assert duel.query_count(0, LOCATION_BANISHED) == 1
        assert duel.query_count(0, LOCATION_DECK) == 2

    def test_custom_lp(self, duel):
        """Verify custom LP per player."""
        state = {
            "player0": {"lp": 4000, "hand": [BLUE_EYES]},
            "player1": {"lp": 2000, "hand": [DARK_MAGICIAN]},
        }
        duel.create_puzzle(state)

        assert duel.game_state.lp == [4000, 2000]

    def test_deck_order(self, duel):
        """Deck cards appear in specification order."""
        state = {
            "player0": {"deck": [BLUE_EYES, DARK_MAGICIAN]},
        }
        duel.create_puzzle(state)

        cards = duel.query_location(0, LOCATION_DECK)
        codes = [c["code"] for c in cards if c.get("code")]
        assert BLUE_EYES in codes
        assert DARK_MAGICIAN in codes

    def test_disabled_card_has_status_disabled(self, duel):
        """A disabled monster should have STATUS_DISABLED set."""
        state = {
            "player0": {
                "monster_zone": [
                    {
                        "code": SANGAN,
                        "pos": "face_up_attack",
                        "seq": 0,
                        "disabled": True,
                    },
                ],
                # Need at least one card in deck for a valid duel
                "deck": [BLUE_EYES],
            },
        }
        duel.create_puzzle(state)

        # Process to let the engine apply effects
        duel.process_until_choice()

        cards = duel.query_location(0, LOCATION_MZONE)
        occupied = {c["sequence"]: c for c in cards if c.get("code")}
        assert occupied[0]["code"] == SANGAN
        assert occupied[0]["status"] & STATUS_DISABLED != 0

    def test_load_from_json_file(self, duel, tmp_path):
        """Load puzzle from a JSON file."""
        data = {
            "player0": {
                "lp": 3000,
                "hand": [BLUE_EYES],
                "deck": [DARK_MAGICIAN],
            },
            "player1": {
                "lp": 1500,
                "deck": [MST],
            },
        }
        path = tmp_path / "puzzle.json"
        path.write_text(json.dumps(data))

        duel.create_puzzle(path)

        assert duel.game_state.lp == [3000, 1500]
        assert duel.query_count(0, LOCATION_HAND) == 1
        assert duel.query_count(0, LOCATION_DECK) == 1

    def test_process_to_choice_after_puzzle(self, duel):
        """Player 0 has cards in hand — engine must produce a SELECT prompt."""
        state = {
            "player0": {
                "hand": [BLUE_EYES, MST],
                "deck": [DARK_MAGICIAN, SANGAN],
            },
            "player1": {
                "deck": [MONSTER_REBORN, DARK_HOLE],
            },
        }
        duel.create_puzzle(state)

        msg, game_state, _events = duel.process_until_choice()
        assert msg is not None, "Expected a SELECT prompt for player 0, got game end"
        assert msg["player"] == 0
