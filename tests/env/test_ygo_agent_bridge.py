"""Tests for yugioh_env.ygo_agent.bridge — obs/msg translation."""

from __future__ import annotations

import numpy as np
import pytest

from yugioh_core.action_categories import (
    BATTLE_ATTACK,
    BATTLE_TO_M2,
    IDLE_ACTIVATE,
    IDLE_SUMMON,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from yugioh_core.constants import (
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
)
from yugioh_core.encoding import CARD_FEATURES, MAX_CARDS, encode_card, encode_u16
from yugioh_env.ygo_agent.bridge import (
    build_predict_input,
    match_response,
    translate_action_msg,
    translate_cards,
    translate_global,
)


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

    def test_skips_empty_mzone_szone_slots(self):
        # The engine reports empty monster/spell zone slots as code==0 rows
        # with a non-zero location byte. These are holes, not cards, and must
        # not be sent to ygo-agent (which never sees empty zones natively).
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        obs[0] = encode_card(
            code=0, location=0x04, sequence=0, position=0, controller=0, is_public=False
        )  # empty mzone slot
        obs[1] = encode_card(
            code=0, location=0x08, sequence=3, position=0, controller=1, is_public=False
        )  # empty szone slot
        obs[2] = encode_card(
            code=89631139, location=0x04, sequence=1, position=0x1, controller=0, is_public=True
        )  # a real monster
        cards = translate_cards(obs)
        assert len(cards) == 1
        assert cards[0]["code"] == 89631139

    def test_keeps_hidden_hand_card(self):
        # A hidden card in hand/deck/extra is code==0 but a REAL card (the
        # model needs the hand/deck counts). Only zone holes are dropped.
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        obs[0] = encode_card(
            code=0, location=0x02, sequence=0, position=0, controller=1, is_public=False
        )  # opponent's hidden hand card
        cards = translate_cards(obs)
        assert len(cards) == 1
        assert cards[0]["location"] == "hand"
        assert cards[0]["controller"] == "opponent"

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


class TestTranslateActionMsg:
    def test_idle_cmd_summon_and_to_bp(self):
        msg = {
            "msg_type": MSG_SELECT_IDLECMD,
            "player": 0,
            "summonable": [
                {"code": 89631139, "controller": 0, "location": 0x02, "sequence": 0},
            ],
            "sp_summonable": [],
            "repositionable": [],
            "mset": [],
            "sset": [],
            "activatable": [],
            "to_bp": 1,
            "to_ep": 1,
            "shuffle_hand": 0,
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_idlecmd"
        cmds = result["data"]["idle_cmds"]
        # summon + to_bp + to_ep = 3 commands
        assert len(cmds) == 3
        assert cmds[0]["cmd_type"] == "summon"
        assert cmds[0]["data"]["card_info"]["code"] == 89631139
        assert cmds[1]["cmd_type"] == "to_bp"
        assert cmds[2]["cmd_type"] == "to_ep"

    def test_idle_cmd_activate_with_desc(self):
        msg = {
            "msg_type": MSG_SELECT_IDLECMD,
            "player": 0,
            "summonable": [],
            "sp_summonable": [],
            "repositionable": [],
            "mset": [],
            "sset": [],
            "activatable": [
                {
                    "code": 12345,
                    "controller": 0,
                    "location": 0x08,
                    "sequence": 1,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
            "to_bp": 0,
            "to_ep": 1,
            "shuffle_hand": 0,
        }
        result = translate_action_msg(msg)
        cmds = result["data"]["idle_cmds"]
        assert cmds[0]["cmd_type"] == "activate"
        assert cmds[0]["data"]["card_info"]["code"] == 12345

    def test_chain_with_cancel(self):
        msg = {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 0,
            "spe_count": 0,
            "hint_timing": 0,
            "other_timing": 0,
            "chains": [
                {
                    "code": 111,
                    "controller": 0,
                    "location": 0x08,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_chain"
        assert result["data"]["forced"] is False
        assert len(result["data"]["chains"]) == 1

    def test_chain_forced(self):
        msg = {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 1,
            "spe_count": 0,
            "hint_timing": 0,
            "other_timing": 0,
            "chains": [
                {
                    "code": 222,
                    "controller": 0,
                    "location": 0x04,
                    "sequence": 1,
                    "position": 0,
                    "desc": 0,
                    "client_mode": 0,
                },
            ],
        }
        result = translate_action_msg(msg)
        assert result["data"]["forced"] is True

    def test_select_effectyn(self):
        msg = {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": 0x04,
            "sequence": 2,
            "position": 0,
            "desc": 89631139 << 4 | 0,
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_effectyn"
        assert result["data"]["code"] == 89631139

    def test_select_yesno(self):
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_yesno"
        assert result["data"]["effect_description"] == 30

    def test_select_position(self):
        msg = {
            "msg_type": MSG_SELECT_POSITION,
            "player": 0,
            "code": 111,
            "positions": 0x5,  # FU-Atk | FU-Def
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_position"
        assert "faceup_attack" in result["data"]["positions"]
        assert "faceup_defense" in result["data"]["positions"]

    def test_select_option(self):
        msg = {"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [1050, 1051]}
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_option"
        assert len(result["data"]["options"]) == 2

    def test_select_card(self):
        msg = {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 1,
            "cards": [
                {"code": 111, "controller": 0, "location": 0x02, "sequence": 0, "subsequence": 0},
                {"code": 222, "controller": 0, "location": 0x02, "sequence": 1, "subsequence": 0},
            ],
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_card"
        assert result["data"]["min"] == 1
        assert len(result["data"]["cards"]) == 2

    def test_select_card_non_overlay_has_overlay_seq_minus_one(self):
        # The 4th loc_info field the parser stores as "subsequence" is actually
        # the card's POSITION bitmask (0x0A = face-down for hand cards), not an
        # Xyz overlay index. A hand/deck card is never an overlay material, so
        # overlay_sequence must be -1; otherwise the server builds a bogus spec
        # ("h1a11") that fails to match the card list and the model can't tell
        # the cards apart.
        msg = {
            "msg_type": MSG_SELECT_CARD,
            "player": 0,
            "cancelable": 0,
            "min": 1,
            "max": 1,
            "cards": [
                # subsequence=10 is the facedown position bitmask, not overlay 10
                {"code": 111, "controller": 0, "location": 0x02, "sequence": 0, "subsequence": 10},
            ],
        }
        result = translate_action_msg(msg)
        assert result["data"]["cards"][0]["location"]["overlay_sequence"] == -1

    def test_battlecmd_attack(self):
        msg = {
            "msg_type": MSG_SELECT_BATTLECMD,
            "player": 0,
            "activatable": [],
            "attackable": [
                {
                    "code": 111,
                    "controller": 0,
                    "location": 0x04,
                    "sequence": 0,
                    "direct_attackable": 0,
                },
            ],
            "to_m2": 1,
            "to_ep": 0,
        }
        result = translate_action_msg(msg)
        assert result["data"]["msg_type"] == "select_battlecmd"
        cmds = result["data"]["battle_cmds"]
        assert cmds[0]["cmd_type"] == "attack"
        assert cmds[1]["cmd_type"] == "to_m2"


class TestBuildPredictInput:
    def test_assembles_all_parts(self):
        obs = {
            "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            "global_state": np.zeros(20, dtype=np.uint8),
        }
        # Set minimal global: turn=1, phase=main1, is_my_turn
        obs["global_state"][4] = 1
        obs["global_state"][5] = 0x04
        obs["global_state"][6] = 1
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        result = build_predict_input(obs, msg, prev_action_idx=0)
        assert "input" in result
        assert "prev_action_idx" in result
        assert "index" in result
        assert result["input"]["global"]["phase"] == "main1"
        assert result["input"]["action_msg"]["data"]["msg_type"] == "select_yesno"

    def test_injects_hidden_deck_cards(self):
        # Deck cards are hidden and absent from the obs card array (only their
        # count lives in global_state). ygo-agent counts deck cards from the
        # card list, so the bridge must synthesize hidden deck placeholders or
        # the model sees an empty deck (off-distribution → uniform policy).
        obs = {
            "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            "global_state": np.zeros(20, dtype=np.uint8),
        }
        obs["global_state"][4] = 1  # turn
        obs["global_state"][5] = 0x04  # phase main1
        obs["global_state"][6] = 1  # is_my_turn
        obs["global_state"][10] = 33  # agent deck count
        obs["global_state"][15] = 36  # opponent deck count
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        result = build_predict_input(obs, msg, prev_action_idx=0)
        cards = result["input"]["cards"]
        my_deck = [c for c in cards if c["location"] == "deck" and c["controller"] == "me"]
        op_deck = [c for c in cards if c["location"] == "deck" and c["controller"] == "opponent"]
        assert len(my_deck) == 33
        assert len(op_deck) == 36
        # Deck cards are hidden: code 0.
        assert all(c["code"] == 0 for c in my_deck + op_deck)


class TestMatchResponse:
    def test_idle_summon(self):
        actions = [
            {"category": IDLE_SUMMON, "index": 0, "code": 111},
            {"category": IDLE_SUMMON, "index": 1, "code": 222},
            {"category": IDLE_TO_BP, "index": 0, "code": 0},
        ]
        # ygo-agent response for summon index 1: (1 << 16) | 0 = 65536
        assert match_response(MSG_SELECT_IDLECMD, actions, (1 << 16) | IDLE_SUMMON) == 1

    def test_idle_activate(self):
        actions = [
            {"category": IDLE_SUMMON, "index": 0, "code": 111},
            {"category": IDLE_ACTIVATE, "index": 0, "code": 222},
            {"category": IDLE_TO_BP, "index": 0, "code": 0},
        ]
        assert match_response(MSG_SELECT_IDLECMD, actions, (0 << 16) | IDLE_ACTIVATE) == 1

    def test_idle_to_bp(self):
        actions = [
            {"category": IDLE_SUMMON, "index": 0, "code": 111},
            {"category": IDLE_TO_BP, "index": 0, "code": 0},
            {"category": IDLE_TO_EP, "index": 0, "code": 0},
        ]
        # ygo-agent response for to_bp is 6
        assert match_response(MSG_SELECT_IDLECMD, actions, 6) == 1

    def test_idle_to_ep(self):
        actions = [
            {"category": IDLE_TO_BP, "index": 0, "code": 0},
            {"category": IDLE_TO_EP, "index": 0, "code": 0},
        ]
        assert match_response(MSG_SELECT_IDLECMD, actions, 7) == 1

    def test_chain_select(self):
        actions = [
            {"category": 0, "index": 0, "code": 111},
            {"category": 0, "index": 1, "code": 222},
            {"category": 1, "index": 0, "code": 0},  # cancel
        ]
        assert match_response(MSG_SELECT_CHAIN, actions, 1) == 1

    def test_chain_cancel(self):
        actions = [
            {"category": 0, "index": 0, "code": 111},
            {"category": 1, "index": 0, "code": 0},
        ]
        assert match_response(MSG_SELECT_CHAIN, actions, -1) == 1

    def test_battle_attack(self):
        actions = [
            {"category": BATTLE_ATTACK, "index": 0, "code": 111},
            {"category": BATTLE_TO_M2, "index": 0, "code": 0},
        ]
        assert match_response(MSG_SELECT_BATTLECMD, actions, (0 << 16) | BATTLE_ATTACK) == 0

    def test_battle_to_m2(self):
        actions = [
            {"category": BATTLE_ATTACK, "index": 0, "code": 111},
            {"category": BATTLE_TO_M2, "index": 0, "code": 0},
        ]
        assert match_response(MSG_SELECT_BATTLECMD, actions, 2) == 1

    def test_yesno_yes(self):
        actions = [
            {"category": 0, "index": 0},
            {"category": 1, "index": 0},
        ]
        assert match_response(MSG_SELECT_YESNO, actions, 1) == 0

    def test_yesno_no(self):
        actions = [
            {"category": 0, "index": 0},
            {"category": 1, "index": 0},
        ]
        assert match_response(MSG_SELECT_YESNO, actions, 0) == 1

    def test_position(self):
        actions = [
            {"category": 0, "index": 0x1},  # faceup_attack
            {"category": 0, "index": 0x4},  # faceup_defense
        ]
        assert match_response(MSG_SELECT_POSITION, actions, 0x4) == 1

    def test_select_card(self):
        actions = [
            {"category": 0, "index": 0, "code": 111},
            {"category": 0, "index": 1, "code": 222},
        ]
        assert match_response(MSG_SELECT_CARD, actions, 1) == 1

    def test_select_card_finish(self):
        # In multi-select, the model returns response=-1 to finish selecting.
        # This must map to the finish action (category==1), not fall back to 0.
        actions = [
            {"category": 0, "index": 1, "code": 111},  # a selectable card
            {"category": 1, "index": 0, "code": 0},  # finish
        ]
        assert match_response(MSG_SELECT_CARD, actions, -1) == 1

    def test_select_tribute_finish(self):
        actions = [
            {"category": 0, "index": 1, "code": 111},
            {"category": 1, "index": 0, "code": 0},  # finish
        ]
        assert match_response(MSG_SELECT_TRIBUTE, actions, -1) == 1

    def test_select_sum_finish(self):
        actions = [
            {"category": 0, "index": 1, "code": 111},
            {"category": 1, "index": 0, "code": 0},  # finish
        ]
        assert match_response(MSG_SELECT_SUM, actions, -1) == 1

    def test_select_option(self):
        actions = [
            {"category": 0, "index": 0},
            {"category": 0, "index": 1},
        ]
        assert match_response(MSG_SELECT_OPTION, actions, 1) == 1

    def test_unselect_card_finish(self):
        actions = [
            {"category": 0, "index": 0, "code": 111},
            {"category": 1, "index": 0, "code": 0},  # finish
        ]
        assert match_response(MSG_SELECT_UNSELECT_CARD, actions, -1) == 1

    def test_no_match_returns_zero(self):
        actions = [{"category": 0, "index": 0, "code": 111}]
        assert match_response(MSG_SELECT_IDLECMD, actions, 99999) == 0
