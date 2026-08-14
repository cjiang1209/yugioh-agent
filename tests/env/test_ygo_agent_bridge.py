"""Tests for yugioh_env.ygo_agent.bridge — observation to ygo-agent translation.

The bridge builds its request from ``YuGiOhObservation.action_descriptors`` and
``prompt_meta``. Outbound bodies are pinned byte-for-byte against
``tests/env/fixtures/ygo_agent_predict_requests.json``, which records the wire contract
the ygo-agent server expects; regenerate it with
``tests/env/fixtures/capture_ygo_agent_predict_requests.py`` only for a deliberate,
reviewed change to that contract. Inbound response matching
(``match_response``) is validated separately via an explicit round-trip table,
since the goldens only exercise the outbound direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.env.conftest import (
    CARD_A,
    CARD_B,
    CARD_C,
    MINIMAL_MSGS,
    MULTI_STEP_CASES,
    obs_from_msg,
)
from yugioh_core.action_categories import (
    BATTLE_ACTIVATE,
    BATTLE_ATTACK,
    BATTLE_TO_EP,
    BATTLE_TO_M2,
    IDLE_ACTIVATE,
    IDLE_SUMMON,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from yugioh_core.constants import (
    ATTRIBUTE_ALL,
    ATTRIBUTE_LIGHT,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_ANNOUNCE_ATTRIB,
    MSG_ANNOUNCE_NUMBER,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    MSG_SELECT_OPTION,
    MSG_SELECT_PLACE,
    MSG_SELECT_POSITION,
    MSG_SELECT_SUM,
    MSG_SELECT_TRIBUTE,
    MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_YESNO,
    PHASE_MAIN1,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
    RACE_DRAGON,
)
from yugioh_core.encoding import (
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_CARDS,
    encode_card,
    encode_u16,
)
from yugioh_env.models import YuGiOhObservation
from yugioh_env.ygo_agent.bridge import (
    _ACTION_MSG_TRANSLATORS,
    build_predict_input,
    match_response,
    translate_action_msg,
    translate_cards,
    translate_global,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Derived, not restated: the goldens must cover exactly what the bridge
# claims to translate.
TRANSLATED_MSG_TYPES = sorted(_ACTION_MSG_TRANSLATORS)


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
                    "location": LOCATION_MZONE,  # mzone
                    "sequence": 2,
                    "position": POS_FACEUP_ATTACK,  # faceup_attack
                    "controller": 0,  # me
                    "is_public": True,
                    "card_type": 0x11,  # monster + normal
                    "level": 8,
                    "attribute": ATTRIBUTE_LIGHT,  # light
                    "race": RACE_DRAGON,  # dragon
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
                    "location": LOCATION_MZONE,  # mzone
                    "sequence": 0,
                    "position": POS_FACEDOWN_DEFENSE,  # facedown_defense
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
                    "location": LOCATION_MZONE,  # mzone
                    "sequence": 1,
                    "position": POS_FACEUP_ATTACK,
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
            code=1,
            location=LOCATION_HAND,
            sequence=0,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
        )
        # slot 1 is empty (all zeros)
        obs[2] = encode_card(
            code=2,
            location=LOCATION_HAND,
            sequence=1,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
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
            code=0, location=LOCATION_MZONE, sequence=0, position=0, controller=0, is_public=False
        )  # empty mzone slot
        obs[1] = encode_card(
            code=0, location=LOCATION_SZONE, sequence=3, position=0, controller=1, is_public=False
        )  # empty szone slot
        obs[2] = encode_card(
            code=89631139,
            location=LOCATION_MZONE,
            sequence=1,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
        )  # a real monster
        cards = translate_cards(obs)
        assert len(cards) == 1
        assert cards[0]["code"] == 89631139

    def test_keeps_hidden_hand_card(self):
        # A hidden card in hand/deck/extra is code==0 but a REAL card (the
        # model needs the hand/deck counts). Only zone holes are dropped.
        obs = np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        obs[0] = encode_card(
            code=0, location=LOCATION_HAND, sequence=0, position=0, controller=1, is_public=False
        )  # opponent's hidden hand card
        cards = translate_cards(obs)
        assert len(cards) == 1
        assert cards[0]["location"] == "hand"
        assert cards[0]["controller"] == "opponent"

    @pytest.mark.parametrize(
        "loc_byte,expected",
        [
            (LOCATION_HAND, "hand"),
            (LOCATION_MZONE, "mzone"),
            (LOCATION_SZONE, "szone"),
            (LOCATION_GRAVE, "grave"),
            (LOCATION_BANISHED, "removed"),
            (LOCATION_EXTRA, "extra"),
        ],
    )
    def test_location_mapping(self, loc_byte, expected):
        obs = self._make_obs_cards(
            [
                {
                    "code": 1,
                    "location": loc_byte,
                    "sequence": 0,
                    "position": POS_FACEUP_ATTACK,
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
                    "location": LOCATION_SZONE,
                    "sequence": 0,
                    "position": POS_FACEUP_ATTACK,
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
        self, my_lp=8000, opp_lp=8000, turn=1, phase=PHASE_MAIN1, is_my_turn=True
    ) -> np.ndarray:
        """Build a (21,) uint8 global_state array."""
        g = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
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
    """Exercises translate_action_msg(descriptors, prompt_meta) via obs_from_msg."""

    def test_idle_cmd_summon_and_to_bp(self):
        msg = {
            "msg_type": MSG_SELECT_IDLECMD,
            "player": 0,
            "summonable": [
                {"code": 89631139, "controller": 0, "location": LOCATION_HAND, "sequence": 0},
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
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
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
                    "location": LOCATION_SZONE,
                    "sequence": 1,
                    "desc": 0x99,
                    "client_mode": 0,
                },
            ],
            "to_bp": 0,
            "to_ep": 1,
            "shuffle_hand": 0,
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        cmds = result["data"]["idle_cmds"]
        assert cmds[0]["cmd_type"] == "activate"
        assert cmds[0]["data"]["card_info"]["code"] == 12345

    def test_chain_with_cancel(self):
        msg = {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 0,
            "chains": [
                {
                    "code": 111,
                    "controller": 0,
                    "location": LOCATION_SZONE,
                    "sequence": 0,
                    "position": 0,
                    "desc": 0x99,
                    "client_mode": 0,
                },
            ],
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_chain"
        assert result["data"]["forced"] is False
        assert len(result["data"]["chains"]) == 1

    def test_chain_forced(self):
        msg = {
            "msg_type": MSG_SELECT_CHAIN,
            "player": 0,
            "forced": 1,
            "chains": [
                {
                    "code": 222,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 1,
                    "position": 0,
                    "desc": 0x99,
                    "client_mode": 0,
                },
            ],
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["forced"] is True

    def test_effectyn_relativizes_controller_at_seat_1(self):
        """Every other bridge fixture runs at agent_player=0 with
        controller=0 cards, where relativization is the identity -- byte-
        equal goldens there don't actually exercise the controller-
        relativization improvement. Here the ygo-agent seat is player 1 and
        the prompt's card is absolute controller 1 (i.e. the ygo-agent
        seat's OWN card), so a correct translation reports "me"."""
        msg = {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 1,
            "code": 89631139,
            "controller": 1,
            "location": LOCATION_MZONE,
            "sequence": 0,
            "desc": 89631139 << 4 | 0,
        }
        obs = obs_from_msg(msg, agent_player=1)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["location"]["controller"] == "me"

    def test_select_effectyn(self):
        msg = {
            "msg_type": MSG_SELECT_EFFECTYN,
            "player": 0,
            "code": 89631139,
            "controller": 0,
            "location": LOCATION_MZONE,
            "sequence": 2,
            "desc": 89631139 << 4 | 0,
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_effectyn"
        assert result["data"]["code"] == 89631139

    def test_select_yesno(self):
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_yesno"
        assert result["data"]["effect_description"] == 30

    def test_select_position(self):
        msg = {
            "msg_type": MSG_SELECT_POSITION,
            "player": 0,
            "code": 111,
            "positions": POS_FACEUP,  # FU-Atk | FU-Def
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_position"
        assert "faceup_attack" in result["data"]["positions"]
        assert "faceup_defense" in result["data"]["positions"]

    def test_select_option(self):
        msg = {"msg_type": MSG_SELECT_OPTION, "player": 0, "options": [1050, 1051]}
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
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
                {
                    "code": 111,
                    "controller": 0,
                    "location": LOCATION_HAND,
                    "sequence": 0,
                    "subsequence": 0,
                },
                {
                    "code": 222,
                    "controller": 0,
                    "location": LOCATION_HAND,
                    "sequence": 1,
                    "subsequence": 0,
                },
            ],
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_card"
        assert result["data"]["min"] == 1
        assert len(result["data"]["cards"]) == 2

    def test_select_unselect_card_cancelable(self):
        # cancelable=1 is a non-default value (the field's fallback is False);
        # this pins _build_prompt_meta actually forwarding it through to the
        # translated body instead of silently dropping to the default.
        msg = {
            "msg_type": MSG_SELECT_UNSELECT_CARD,
            "player": 0,
            "finishable": 1,
            "cancelable": 1,
            "min": 0,
            "max": 1,
            "selectable": [
                {
                    "code": 111,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "subsequence": 0,
                },
            ],
            "unselectable": [],
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_unselect_card"
        assert result["data"]["cancelable"] is True

    def test_announce_attrib_count(self):
        # count=5 is a non-default value (the field's fallback is 1); this
        # pins _build_prompt_meta actually populating `count` for
        # MSG_ANNOUNCE_ATTRIB instead of falling back to the default.
        msg = {"msg_type": MSG_ANNOUNCE_ATTRIB, "player": 0, "count": 5, "available": ATTRIBUTE_ALL}
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "announce_attrib"
        assert result["data"]["count"] == 5

    def test_battlecmd_attack(self):
        msg = {
            "msg_type": MSG_SELECT_BATTLECMD,
            "player": 0,
            "activatable": [],
            "attackable": [
                {
                    "code": 111,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "direct_attackable": 0,
                },
            ],
            "to_m2": 1,
            "to_ep": 0,
        }
        obs = obs_from_msg(msg)
        result = translate_action_msg(obs.action_descriptors, obs.prompt_meta)
        assert result["data"]["msg_type"] == "select_battlecmd"
        cmds = result["data"]["battle_cmds"]
        assert cmds[0]["cmd_type"] == "attack"
        assert cmds[1]["cmd_type"] == "to_m2"


class TestBuildPredictInput:
    def test_assembles_all_parts(self):
        global_state = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
        global_state[4] = 1  # turn
        global_state[5], global_state[6] = encode_u16(PHASE_MAIN1)
        global_state[7] = 1  # is_my_turn
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        base = obs_from_msg(msg)
        obs = YuGiOhObservation(
            cards=np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            global_state=global_state,
            actions=base.actions,
            action_mask=base.action_mask,
            action_descriptors=base.action_descriptors,
            prompt_meta=base.prompt_meta,
        )
        result = build_predict_input(obs, prev_action_idx=0)
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
        global_state = np.zeros(GLOBAL_FEATURES, dtype=np.uint8)
        global_state[4] = 1  # turn
        global_state[5], global_state[6] = encode_u16(PHASE_MAIN1)
        global_state[7] = 1  # is_my_turn
        global_state[10] = 33  # agent deck count
        global_state[15] = 36  # opponent deck count
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        base = obs_from_msg(msg)
        obs = YuGiOhObservation(
            cards=np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            global_state=global_state,
            actions=base.actions,
            action_mask=base.action_mask,
            action_descriptors=base.action_descriptors,
            prompt_meta=base.prompt_meta,
        )
        result = build_predict_input(obs, prev_action_idx=0)
        cards = result["input"]["cards"]
        my_deck = [c for c in cards if c["location"] == "deck" and c["controller"] == "me"]
        op_deck = [c for c in cards if c["location"] == "deck" and c["controller"] == "opponent"]
        assert len(my_deck) == 33
        assert len(op_deck) == 36
        # Deck cards are hidden: code 0.
        assert all(c["code"] == 0 for c in my_deck + op_deck)


# ---------------------------------------------------------------------------
# Golden byte-equality: the outbound wire contract, one body per translated
# msg type plus the mid-selection cases MINIMAL_MSGS cannot express.
# ---------------------------------------------------------------------------

with open(FIXTURES_DIR / "ygo_agent_predict_requests.json") as _f:
    GOLDEN: dict[str, dict] = json.load(_f)


@pytest.mark.parametrize("msg_type", TRANSLATED_MSG_TYPES)
def test_predict_body_byte_equal_after_resourcing(msg_type):
    raw_msg = {**MINIMAL_MSGS[msg_type], "msg_type": msg_type}
    obs = obs_from_msg(raw_msg)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN[str(msg_type)]


# ---------------------------------------------------------------------------
# Response round-trip: match_response is the inverse direction of
# translate_action_msg and is never exercised by the golden bodies above (the
# server's chosen `response` int has no corresponding field in the outbound
# schema for several msg types, e.g. phase/place/confirm). Every case below
# targets a non-zero slot: match_response falls back to slot 0 on no match
# (bridge.py's `_find` returning 0), so a slot-0 expectation would be
# indistinguishable from total match failure.
# ---------------------------------------------------------------------------


# Each fixture below deliberately has >=2 entries in the relevant category so
# that engine_index=1 (or the second bit/option/etc.) lands on a non-zero
# absolute slot in the full descriptor list.
RESPONSE_FIXTURES: dict[str, dict] = {
    "idlecmd": {
        "msg_type": MSG_SELECT_IDLECMD,
        "player": 0,
        "summonable": [CARD_A, CARD_B],
        "sp_summonable": [],
        "repositionable": [],
        "mset": [],
        "sset": [],
        "activatable": [
            {**CARD_A, "desc": 0x10, "client_mode": 0},
            {**CARD_B, "desc": 0x20, "client_mode": 0},
        ],
        "to_bp": 1,
        "to_ep": 1,
        "shuffle_hand": 0,
    },
    "battlecmd": {
        "msg_type": MSG_SELECT_BATTLECMD,
        "player": 0,
        "activatable": [
            {**CARD_A, "desc": 0x10, "client_mode": 0},
            {**CARD_B, "desc": 0x20, "client_mode": 0},
        ],
        "attackable": [
            {**CARD_A, "direct_attackable": 0},
            {**CARD_B, "direct_attackable": 1},
        ],
        "to_m2": 1,
        "to_ep": 1,
    },
    "chain": {
        "msg_type": MSG_SELECT_CHAIN,
        "player": 0,
        "forced": 0,
        "chains": [
            {**CARD_A, "desc": 0x10, "position": 0},
            {**CARD_B, "desc": 0x20, "position": 0},
        ],
    },
    "card": {
        "msg_type": MSG_SELECT_CARD,
        "player": 0,
        "cancelable": 0,
        "min": 0,
        "max": 1,
        "cards": [CARD_A, CARD_B],
    },
    "tribute": {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": 0,
        "cancelable": 0,
        "min": 1,
        "max": 1,
        "cards": [{**CARD_A, "release_param": 1}, {**CARD_B, "release_param": 1}],
    },
    "sum": {
        "msg_type": MSG_SELECT_SUM,
        "player": 0,
        "select_type": 0,
        "target_sum": 4,
        "min": 1,
        "max": 1,
        "must_cards": [],
        "optional_cards": [{**CARD_A, "param": 4}, {**CARD_B, "param": 4}],
    },
    "unselect": {
        "msg_type": MSG_SELECT_UNSELECT_CARD,
        "player": 0,
        "finishable": 1,
        "cancelable": 0,
        "min": 0,
        "max": 1,
        "selectable": [{**CARD_A, "subsequence": 0}, {**CARD_B, "subsequence": 0}],
        "unselectable": [],
    },
    "option": {
        "msg_type": MSG_SELECT_OPTION,
        "player": 0,
        "options": [7, 8],
    },
    "position": {
        "msg_type": MSG_SELECT_POSITION,
        "player": 0,
        "code": 111,
        "positions": 0x0F,
    },
    "place": {
        "msg_type": MSG_SELECT_PLACE,
        "player": 0,
        "count": 1,
        "field_mask": 0,
    },
    "announce_attrib": {
        "msg_type": MSG_ANNOUNCE_ATTRIB,
        "player": 0,
        "count": 1,
        "available": 0x03,
    },
    "announce_number": {
        "msg_type": MSG_ANNOUNCE_NUMBER,
        "player": 0,
        "numbers": [4, 8],
    },
    "effectyn": {
        "msg_type": MSG_SELECT_EFFECTYN,
        "player": 0,
        "code": 111,
        "controller": 0,
        "location": LOCATION_MZONE,
        "sequence": 0,
        "desc": 0x99,
    },
    "yesno": {
        "msg_type": MSG_SELECT_YESNO,
        "player": 0,
        "desc": 0x99,
    },
}

# (msg_type, fixture_key, expected_slot, server_response)
# Slots follow ActionMapper's extraction order.
RESPONSE_CASES: list[tuple[int, str, int, int]] = [
    # IDLECMD: card_command (summon idx1), activate_effect (tagged IDLE_ACTIVATE=5),
    # phase_change (to_bp/to_ep raw category values 6/7).
    (MSG_SELECT_IDLECMD, "idlecmd", 1, (1 << 16) | IDLE_SUMMON),
    (MSG_SELECT_IDLECMD, "idlecmd", 3, (1 << 16) | IDLE_ACTIVATE),
    (MSG_SELECT_IDLECMD, "idlecmd", 4, IDLE_TO_BP),
    (MSG_SELECT_IDLECMD, "idlecmd", 5, IDLE_TO_EP),
    # BATTLECMD: activate_effect (tagged BATTLE_ACTIVATE=0), attack (tagged
    # BATTLE_ATTACK=1), phase_change (to_m2/to_ep raw category values 2/3).
    (MSG_SELECT_BATTLECMD, "battlecmd", 1, (1 << 16) | BATTLE_ACTIVATE),
    (MSG_SELECT_BATTLECMD, "battlecmd", 3, (1 << 16) | BATTLE_ATTACK),
    (MSG_SELECT_BATTLECMD, "battlecmd", 4, BATTLE_TO_M2),
    (MSG_SELECT_BATTLECMD, "battlecmd", 5, BATTLE_TO_EP),
    # CHAIN: activate_effect matches on the PLAIN engine_index (no tag);
    # pass is -1.
    (MSG_SELECT_CHAIN, "chain", 1, 1),
    (MSG_SELECT_CHAIN, "chain", 2, -1),
    # CARD/TRIBUTE/SUM: pick_card matches on engine_index; CARD's finish is -1.
    (MSG_SELECT_CARD, "card", 1, 1),
    (MSG_SELECT_CARD, "card", 2, -1),
    (MSG_SELECT_TRIBUTE, "tribute", 1, 1),
    (MSG_SELECT_SUM, "sum", 1, 1),
    # UNSELECT: pick_card matches on engine_index; its own pass is -1 (a
    # DIFFERENT matcher function than CARD/TRIBUTE/SUM's finish handling).
    (MSG_SELECT_UNSELECT_CARD, "unselect", 1, 1),
    (MSG_SELECT_UNSELECT_CARD, "unselect", 2, -1),
    # OPTION: choose_option matches on engine_index.
    (MSG_SELECT_OPTION, "option", 1, 1),
    # POSITION: choose_position matches on the position BITMASK (0x4 =
    # faceup_defense), not a list ordinal.
    (MSG_SELECT_POSITION, "position", 2, POS_FACEUP_DEFENSE),
    # PLACE: place_zone's response is the ordinal position in the outbound
    # list, which (PLACE prompts contain only place_zone descriptors,
    # contiguous from slot 0) equals the absolute slot.
    (MSG_SELECT_PLACE, "place", 2, 2),
    # ANNOUNCE_ATTRIB: pick_bit matches on `value` (the 1<<bit MASK), NOT
    # `engine_index` (the bit number). bit=1 -> mask=2; if a bug matched on
    # engine_index instead, response=2 would find nothing and fall back to 0.
    (MSG_ANNOUNCE_ATTRIB, "announce_attrib", 1, 2),
    # ANNOUNCE_NUMBER: announce_number matches on engine_index (not `value`,
    # the announced number itself).
    (MSG_ANNOUNCE_NUMBER, "announce_number", 1, 1),
    # EFFECTYN/YESNO: confirm matches on 1=yes/0=no. Both share
    # _match_confirm_response; testing "no" (response=0) lands on slot 1
    # since slot 0 is always the "yes" descriptor.
    (MSG_SELECT_EFFECTYN, "effectyn", 1, 0),
    (MSG_SELECT_YESNO, "yesno", 1, 0),
]


def _fixture_obs(key: str) -> YuGiOhObservation:
    return obs_from_msg(RESPONSE_FIXTURES[key])


@pytest.mark.parametrize("msg_type,fixture_key,slot,response", RESPONSE_CASES)
def test_response_round_trip(msg_type, fixture_key, slot, response):
    obs = _fixture_obs(fixture_key)
    assert match_response(msg_type, obs.action_descriptors, response) == slot


@pytest.mark.parametrize("msg_type,fixture_key,slot,response", RESPONSE_CASES)
def test_response_fixtures_expose_nonzero_slots(msg_type, fixture_key, slot, response):
    """Guard against a vacuous round-trip: the expected slot must actually be
    a legal (masked-in) action, and non-zero, or the parametrization above
    can't distinguish a real match from match_response's slot-0 fallback."""
    obs = _fixture_obs(fixture_key)
    assert slot != 0
    assert slot < int(obs.action_mask.sum())


def test_match_response_unknown_msg_type_falls_back_to_zero():
    obs = _fixture_obs("idlecmd")
    assert match_response(999999, obs.action_descriptors, 0) == 0


def test_match_response_no_match_falls_back_to_zero():
    obs = _fixture_obs("idlecmd")
    # A response value that can't match anything in this fixture.
    assert match_response(MSG_SELECT_IDLECMD, obs.action_descriptors, 0xDEADBEEF) == 0


# ---------------------------------------------------------------------------
# Mid-selection goldens: the cases above all have an EMPTY selection, so none
# exercise the compaction of already-picked entries out of the descriptor
# list. That compaction is what makes a card's POSITION in the list we send
# diverge from its ENGINE INDEX, which ygoinf reads as positionally aligned
# with `selected` (see bridge.py's `_reconstruct_pick_list`).
# ---------------------------------------------------------------------------


def test_predict_body_select_card_mid_selection_byte_equal():
    """All 3 cards are sent at their engine-index slots with selected=[1],
    though the descriptor list offers only 0 and 2."""
    msg, selected = MULTI_STEP_CASES["card_mid_select"]
    obs = obs_from_msg(msg, _selected=selected)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN["card_mid_select"]


def test_predict_body_select_tribute_mid_selection_byte_equal():
    """2 cards, index 0 already picked toward a 2-tribute release."""
    msg, selected = MULTI_STEP_CASES["tribute_mid_select"]
    obs = obs_from_msg(msg, _selected=selected)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN["tribute_mid_select"]


def test_predict_body_select_sum_mid_selection_byte_equal():
    """3 equal-level optional cards, index 0 already picked. Levels are
    chosen so every remaining card stays `reachable` toward target_sum=8 —
    isolating the compaction fix from the separate (intentional) reachable-
    pruning behavior of `_extract_sum_actions`, which the raw pre-refactor
    bridge never applied at all."""
    msg, selected = MULTI_STEP_CASES["sum_mid_select"]
    obs = obs_from_msg(msg, _selected=selected)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN["sum_mid_select"]


@pytest.mark.parametrize(
    "levels",
    [
        [2, 5, 6],  # reviewer's crash repro
        [4, 3, 4],  # reviewer's second crash repro
    ],
)
def test_predict_body_select_sum_pruned_gap_does_not_crash(levels):
    """Regression for the Critical bug: ``_extract_sum_actions``'s
    ``reachable`` filter (action_space.py) prunes optional cards that can't
    participate in any valid completion of the sum. With levels[1]
    unreachable toward target_sum=8, the descriptor list has engine indices
    [0, 2] but not 1 -- a genuine, non-contiguous gap that a pre-fix
    contiguity assertion in ``_reconstruct_pick_list`` mistook for a bug and
    raised on. It's a routine level-sum board, not a corrupted engine
    response; ``build_predict_input`` must handle it without raising, and
    the pruned index must simply be absent from the outbound `cards`."""
    msg = {
        "msg_type": MSG_SELECT_SUM,
        "player": 0,
        "select_type": 0,
        "target_sum": 8,
        "min": 1,
        "max": 3,
        "must_cards": [],
        "optional_cards": [
            {**CARD_A, "param": levels[0]},
            {**CARD_B, "param": levels[1]},
            {**CARD_C, "param": levels[2]},
        ],
    }
    obs = obs_from_msg(msg)
    body = build_predict_input(obs, prev_action_idx=0)  # must not raise
    responses = [c["response"] for c in body["input"]["action_msg"]["data"]["cards"]]
    assert responses == [0, 2]
    assert body["input"]["action_msg"]["data"]["selected"] == []


@pytest.mark.parametrize("key", MULTI_STEP_CASES)
def test_mid_selection_goldens_record_their_picks(key):
    """A case whose golden lost its picks would silently stop exercising the
    reconstruction while still passing byte-equality. Keyed off the body:
    only the pick-list prompts emit ``selected`` at all."""
    _msg, selected = MULTI_STEP_CASES[key]
    assert selected, "a mid-selection case must have picks"
    data = GOLDEN[key]["input"]["action_msg"]["data"]
    if "selected" in data:
        assert len(data["selected"]) == len(selected)


def test_golden_case_set_matches_the_tables():
    """The fixture and the tables must describe the same cases.

    Only the subset direction is otherwise checked (a missing golden raises
    KeyError in the parametrized test); an orphan entry -- a msg type moved
    into _SERVER_UNSUPPORTED_MSGS, say -- would linger unnoticed.
    """
    expected = {str(m) for m in TRANSLATED_MSG_TYPES} | set(MULTI_STEP_CASES)
    assert set(GOLDEN) == expected


def test_predict_body_sum_pruned_mid_selection_byte_equal():
    """Card 0 is unreachable once card 1 is picked, so it is pruned and the
    list we send carries engine indices [1, 2] -- the pick at engine 1 lands
    at position 0. `selected` must carry that POSITION; each entry's
    `response` carries the engine index. The assertion re-derives the pairing."""
    msg, selected = MULTI_STEP_CASES["sum_pruned_mid_select"]
    obs = obs_from_msg(msg, _selected=selected)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN["sum_pruned_mid_select"]
    data = body["input"]["action_msg"]["data"]
    assert [data["cards"][p]["response"] for p in data["selected"]] == selected


def test_predict_body_select_card_two_picked_byte_equal():
    """Two picks, in non-identity order (2 then 0). With a single pick any
    permutation of the selected/picked pairing gives the same result, so this
    is the minimum case that pins the order."""
    msg, selected = MULTI_STEP_CASES["card_two_picked"]
    obs = obs_from_msg(msg, _selected=selected)
    body = build_predict_input(obs, prev_action_idx=0)
    assert body == GOLDEN["card_two_picked"]


def test_match_response_select_card_mid_selection_round_trip():
    """Round-trip proof: with card index 1 already picked, the server's
    `response=2` (picking engine index 2, the other still-open card) must
    resolve to the descriptor at that engine index, not to some compacted
    position."""
    msg, selected = MULTI_STEP_CASES["card_mid_select"]
    obs = obs_from_msg(msg, _selected=selected)
    slot = match_response(MSG_SELECT_CARD, obs.action_descriptors, 2)
    d = obs.action_descriptors[slot]
    assert d.__class__.__name__ == "PickCard"
    assert d.engine_index == 2
    assert d.card.code == CARD_C["code"]
