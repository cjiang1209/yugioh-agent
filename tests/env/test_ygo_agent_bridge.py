"""Tests for yugioh_env.ygo_agent.bridge — obs/msg translation."""

from __future__ import annotations

import numpy as np
import pytest

from yugioh_core.encoding import CARD_FEATURES, MAX_CARDS, encode_card, encode_u16
from yugioh_env.ygo_agent.bridge import translate_cards, translate_global


class TestTranslateCards:
    def _make_obs_cards(self, cards_data: list[dict]) -> np.ndarray:
        """Build a (MAX_CARDS, CARD_FEATURES) uint8 array from card specs."""
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        for i, card in enumerate(cards_data):
            obs[i] = encode_card(**card)
        return obs

    def test_empty_obs_returns_empty(self):
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        assert translate_cards(obs) == []

    def test_single_card_basic_fields(self):
        obs = self._make_obs_cards(
            [
                {
                    "code": 89631139,  # Blue-Eyes White Dragon
                    "location": 0x04,  # mzone
                    "sequence": 2,
                    "position": 0x1,  # faceup_attack
                    "controller": 0,  # me
                    "is_public": True,
                    "card_type": 0x11,  # monster + normal
                    "level": 8,
                    "attribute": 0x10,  # light
                    "race": 0x2000,  # dragon
                    "attack": 3000,
                    "defense": 2500,
                    "counter_count": 0,
                    "negated": False,
                }
            ]
        )
        cards = translate_cards(obs)
        assert len(cards) == 1
        c = cards[0]
        assert c["code"] == 89631139
        assert c["location"] == "mzone"
        assert c["sequence"] == 2
        assert c["controller"] == "me"
        assert c["position"] == "faceup_attack"
        assert c["overlay_sequence"] == -1
        assert c["attribute"] == "light"
        assert c["race"] == "dragon"
        assert c["level"] == 8
        assert c["counter"] == 0
        assert c["negated"] is False
        assert c["attack"] == 3000
        assert c["defense"] == 2500
        assert "monster" in c["types"]
        assert "normal" in c["types"]

    def test_opponent_facedown_defense(self):
        obs = self._make_obs_cards(
            [
                {
                    "code": 46986414,
                    "location": 0x04,  # mzone
                    "sequence": 0,
                    "position": 0x8,  # facedown_defense
                    "controller": 1,  # opponent
                    "is_public": False,
                }
            ]
        )
        cards = translate_cards(obs)
        assert len(cards) == 1
        assert cards[0]["controller"] == "opponent"
        assert cards[0]["position"] == "facedown_defense"

    def test_overlay_card(self):
        obs = self._make_obs_cards(
            [
                {
                    "code": 84013237,
                    "location": 0x04,  # mzone
                    "sequence": 1,
                    "position": 0x1,
                    "controller": 0,
                    "is_public": True,
                    "is_overlay": True,
                }
            ]
        )
        cards = translate_cards(obs)
        assert cards[0]["overlay_sequence"] == 0

    def test_multiple_cards_skips_empty_slots(self):
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        obs[0] = encode_card(
            code=1, location=0x02, sequence=0, position=0x1, controller=0, is_public=True
        )
        # slot 1 is empty (all zeros)
        obs[2] = encode_card(
            code=2, location=0x02, sequence=1, position=0x1, controller=0, is_public=True
        )
        cards = translate_cards(obs)
        assert len(cards) == 2
        assert cards[0]["code"] == 1
        assert cards[1]["code"] == 2

    @pytest.mark.parametrize(
        "loc_byte,expected",
        [
            (0x02, "hand"),
            (0x04, "mzone"),
            (0x08, "szone"),
            (0x10, "grave"),
            (0x20, "removed"),
            (0x40, "extra"),
        ],
    )
    def test_location_mapping(self, loc_byte, expected):
        obs = self._make_obs_cards(
            [
                {
                    "code": 1,
                    "location": loc_byte,
                    "sequence": 0,
                    "position": 0x1,
                    "controller": 0,
                    "is_public": True,
                }
            ]
        )
        assert translate_cards(obs)[0]["location"] == expected

    def test_type_flags_decoded(self):
        # spell + continuous = 0x2 | 0x20000 = 0x20002
        obs = self._make_obs_cards(
            [
                {
                    "code": 1,
                    "location": 0x08,
                    "sequence": 0,
                    "position": 0x1,
                    "controller": 0,
                    "is_public": True,
                    "card_type": 0x20002,
                }
            ]
        )
        types = translate_cards(obs)[0]["types"]
        assert "spell" in types
        assert "continuous" in types
        assert "monster" not in types


class TestTranslateGlobal:
    def _make_obs_global(
        self, my_lp=8000, opp_lp=8000, turn=1, phase=0x04, is_my_turn=True
    ) -> np.ndarray:
        """Build a (21,) uint8 global_state array."""
        g = np.zeros(21, dtype=np.uint8)
        g[0], g[1] = encode_u16(my_lp)
        g[2], g[3] = encode_u16(opp_lp)
        g[4] = turn
        g[5], g[6] = encode_u16(phase)
        g[7] = 1 if is_my_turn else 0
        return g

    def test_basic_global(self):
        g = translate_global(self._make_obs_global())
        assert g["my_lp"] == 8000
        assert g["op_lp"] == 8000
        assert g["turn"] == 1
        assert g["phase"] == "main1"
        assert g["is_my_turn"] is True
        assert g["is_first"] is True  # turn 1, my turn → I went first

    def test_second_player(self):
        g = translate_global(self._make_obs_global(turn=1, is_my_turn=False))
        assert g["is_first"] is False
        assert g["is_my_turn"] is False

    def test_lp_values(self):
        g = translate_global(self._make_obs_global(my_lp=4500, opp_lp=12000))
        assert g["my_lp"] == 4500
        assert g["op_lp"] == 12000

    def test_phase_mapping(self):
        for phase_val, expected in [
            (0x01, "draw"),
            (0x02, "standby"),
            (0x04, "main1"),
            (0x08, "battle_start"),
            (0x10, "battle_step"),
            (0x20, "damage"),
            (0x40, "damage_calculation"),
            (0x80, "battle"),
            (0x100, "main2"),
            (0x200, "end"),
        ]:
            g = translate_global(self._make_obs_global(phase=phase_val))
            assert g["phase"] == expected
