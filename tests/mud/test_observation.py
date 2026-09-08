"""Unit tests for MUDObservationBuilder.

Pure unit tests — no torch, no MUD server. Uses CardDatabase from a temp
SQLite DB (same pattern as test_mud_game_state.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.action_categories import (
    BATTLE_ATTACK,
    BATTLE_TO_EP,
    IDLE_ACTIVATE,
    IDLE_SUMMON,
    IDLE_TO_BP,
    IDLE_TO_EP,
)
from core.card_database import CardDatabase
from core.constants import (
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_SELECT_BATTLECMD,
    MSG_SELECT_CARD,
    MSG_SELECT_EFFECTYN,
    MSG_SELECT_IDLECMD,
    PHASE_DRAW,
    PHASE_MAIN1,
    PHASE_MAIN2,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP_ATTACK,
)
from core.encoding import (
    ACTION_FEATURES,
    ACTION_LAYOUT,
    CARD_FEATURES,
    CARD_LAYOUT,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from env.models import YuGiOhObservation
from mud.cmd_handler import StructuredAction
from mud.game_state import CardEntry, MUDGameState
from mud.observation import (
    PHASE_MAP,
    POSITION_MAP,
    PROMPT_MSG_MAP,
    MUDObservationBuilder,
)
from mud.text_parser import ParsedPrompt, PromptType
from rl.obs_encoder import encode_observation
from tests.mud.conftest import build_cards_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Card data for the temp DB
_CARDS = [
    # (id, name, type, atk, def, level, race, attribute)
    (89631139, "Blue-Eyes White Dragon", 17, 3000, 2500, 8, 0x20, 0x10),
    (46986414, "Dark Magician", 17, 2500, 2100, 7, 0x2, 0x20),
    (40640057, "Kuriboh", 17, 300, 200, 1, 0x10, 0x20),
    (44095762, "Mirror Force", 4, 0, 0, 0, 0, 0),
]


@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_cards_db(tmp_path_factory.mktemp("cards") / "cards.cdb", _CARDS)


@pytest.fixture(scope="module")
def card_db(tmp_db: Path) -> CardDatabase:
    return CardDatabase(tmp_db)


@pytest.fixture
def builder(card_db: CardDatabase) -> MUDObservationBuilder:
    return MUDObservationBuilder(card_db)


@pytest.fixture
def gs() -> MUDGameState:
    return MUDGameState()


# ---------------------------------------------------------------------------
# 1. Phase/position/prompt mapping completeness
# ---------------------------------------------------------------------------


class TestMappings:
    def test_phase_map_keys(self):
        assert "draw phase" in PHASE_MAP
        assert "standby phase" in PHASE_MAP
        assert "main1 phase" in PHASE_MAP
        assert "battle phase" in PHASE_MAP
        assert "main2 phase" in PHASE_MAP
        assert "end phase" in PHASE_MAP

    def test_phase_map_values(self):
        assert PHASE_MAP["draw phase"] == PHASE_DRAW
        assert PHASE_MAP["main1 phase"] == PHASE_MAIN1

    def test_position_map_keys(self):
        assert "face-up attack" in POSITION_MAP
        assert "face-down defense" in POSITION_MAP
        assert "face-up defense" in POSITION_MAP
        assert "face-down attack" in POSITION_MAP

    def test_position_map_values(self):
        assert POSITION_MAP["face-up attack"] == POS_FACEUP_ATTACK
        assert POSITION_MAP["face-down defense"] == POS_FACEDOWN_DEFENSE

    def test_prompt_msg_map_idle(self):
        assert PROMPT_MSG_MAP[PromptType.IDLE_CMD] == MSG_SELECT_IDLECMD

    def test_prompt_msg_map_battle(self):
        assert PROMPT_MSG_MAP[PromptType.BATTLE_MENU] == MSG_SELECT_BATTLECMD

    def test_prompt_msg_map_effectyn(self):
        assert PROMPT_MSG_MAP[PromptType.SELECT_EFFECTYN] == MSG_SELECT_EFFECTYN


# ---------------------------------------------------------------------------
# 2. Global state encoding
# ---------------------------------------------------------------------------


class TestGlobalState:
    def test_global_state_matches_the_engine_encoder(self, builder, gs):
        """MUD's global_state must be byte-identical to the engine's.

        The MUD observation is fed to a network trained on engine
        observations, so the two encoders have to agree slot for slot.
        Asserting against `build_observation` rather than re-walking the
        builder's own field order is what makes this a contract test: a
        layout change on either side shows up here instead of being mirrored
        into the expectation.

        `chain_count` and `is_finished` are excluded -- the MUD text parser
        cannot observe them and always writes 0 -- so the engine state below
        sets both to their zero value to keep the arrays comparable.
        """
        from env.game_state import GameState
        from env.observation import build_observation

        gs.my_lp, gs.opp_lp = 7500, 6000
        gs.turn = 3
        gs.phase = "main2 phase"  # 0x100 -- unrepresentable in a single byte
        gs.is_my_turn = True
        gs.my_deck_count = 30
        gs.my_hand = [CardEntry(name="A", code=1)]
        gs.my_graveyard = [CardEntry(name="B"), CardEntry(name="C")]
        gs.my_banished = [CardEntry(name="D")]
        gs.my_extra = []
        gs.opp_deck_count = 28
        gs.opp_hand_count = 5
        gs.opp_graveyard = [CardEntry(name="E")]
        gs.opp_banished = []
        gs.opp_extra = [CardEntry(name="F")]

        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"])
        mud_global = builder.build(gs, prompt)["global_state"]

        engine = GameState()
        engine.lp = [7500, 6000]
        engine.turn_count = 3
        engine.phase = PHASE_MAIN2
        engine.current_player = 0  # agent's turn
        engine.chain_count = 0
        engine.deck_count = [30, 28]
        engine.hand_count = [1, 5]
        engine.grave_count = [2, 1]
        engine.banished_count = [1, 0]
        engine.extra_count = [0, 1]
        engine.is_finished = False
        engine_obs = YuGiOhObservation(
            global_state=build_observation(
                engine, {"msg_type": MSG_SELECT_IDLECMD}, agent_player=0
            )["global_state"]
        )
        engine_global = encode_observation(engine_obs)["global_state"]

        assert mud_global.shape == engine_global.shape == (GLOBAL_FEATURES,)
        mismatches = [i for i in range(len(engine_global)) if mud_global[i] != engine_global[i]]
        assert not mismatches, (
            f"global_state differs from the engine at {mismatches}: "
            f"mud={[int(mud_global[i]) for i in mismatches]} "
            f"engine={[int(engine_global[i]) for i in mismatches]}"
        )


# ---------------------------------------------------------------------------
# 3. Card encoding — known card in hand
# ---------------------------------------------------------------------------


class TestCardEncoding:
    def test_my_hand_card(self, builder, gs):
        gs.my_hand = [
            CardEntry(name="Blue-Eyes", code=89631139, position=""),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"])
        obs = builder.build(gs, prompt)
        c = obs["cards"]

        assert c.shape == (MAX_CARDS, CARD_FEATURES)
        # First card should be the hand card
        card0 = c[0]
        assert CARD_LAYOUT.read(card0, "code") == 89631139
        assert CARD_LAYOUT.read(card0, "location") == LOCATION_HAND
        assert CARD_LAYOUT.read(card0, "sequence") == 0
        assert CARD_LAYOUT.read(card0, "controller") == 0  # agent
        assert CARD_LAYOUT.read(card0, "is_public") == 1
        assert CARD_LAYOUT.read(card0, "attack") == 3000
        assert CARD_LAYOUT.read(card0, "defense") == 2500


# ---------------------------------------------------------------------------
# 4. Opponent hand — hidden entries
# ---------------------------------------------------------------------------


class TestOpponentHand:
    def test_opp_hand_hidden_entries(self, builder, gs):
        gs.opp_hand_count = 3
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"])
        obs = builder.build(gs, prompt)
        c = obs["cards"]

        # Agent has no cards, so opp hand starts at idx 0
        # but agent zones come first (all empty), so opp hand starts
        # after agent zones
        # Find first non-zero card
        opp_hand_cards = []
        for i in range(MAX_CARDS):
            if (
                CARD_LAYOUT.read(c[i], "location") == LOCATION_HAND
                and CARD_LAYOUT.read(c[i], "controller") == 1
            ):
                opp_hand_cards.append(c[i])

        assert len(opp_hand_cards) == 3
        for card in opp_hand_cards:
            assert CARD_LAYOUT.read(card, "code") == 0  # hidden
            assert CARD_LAYOUT.read(card, "controller") == 1  # opponent
            assert CARD_LAYOUT.read(card, "is_public") == 0


# ---------------------------------------------------------------------------
# 5. Opponent face-down
# ---------------------------------------------------------------------------


class TestOpponentFaceDown:
    def test_opp_facedown_monster(self, builder, gs):
        gs.opp_mzone = [
            CardEntry(name="", code=0, position="face-down defense"),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"])
        obs = builder.build(gs, prompt)
        c = obs["cards"]

        # Find the opp mzone card
        opp_mon = []
        for i in range(MAX_CARDS):
            if (
                CARD_LAYOUT.read(c[i], "location") == LOCATION_MZONE
                and CARD_LAYOUT.read(c[i], "controller") == 1
            ):
                opp_mon.append(c[i])

        assert len(opp_mon) == 1
        card = opp_mon[0]
        assert CARD_LAYOUT.read(card, "code") == 0  # hidden
        assert CARD_LAYOUT.read(card, "position") == 0  # face-down
        assert CARD_LAYOUT.read(card, "is_public") == 0


# ---------------------------------------------------------------------------
# 6. Zone fill order
# ---------------------------------------------------------------------------


class TestZoneFillOrder:
    def test_cards_in_rl_order(self, builder, gs):
        gs.my_hand = [CardEntry(name="A", code=40640057)]
        gs.my_mzone = [CardEntry(name="B", code=89631139, position="face-up attack")]
        gs.my_graveyard = [CardEntry(name="C", code=46986414)]

        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"])
        obs = builder.build(gs, prompt)
        c = obs["cards"]

        # Cards should appear in order: hand, mzone, szone, grave, ...
        codes = []
        for i in range(MAX_CARDS):
            code = CARD_LAYOUT.read(c[i], "code")
            if code != 0:
                codes.append(code)

        # hand (Kuriboh), mzone (BEWD), grave (DM)
        assert codes[0] == 40640057  # hand
        assert codes[1] == 89631139  # mzone
        assert codes[2] == 46986414  # grave


# ---------------------------------------------------------------------------
# 7. Idle action features
# ---------------------------------------------------------------------------


class TestIdleActionFeatures:
    def test_idle_structured_actions(self, builder, gs):
        sa = [
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h1",
                card_code=89631139,
                location=LOCATION_HAND,
                sequence=0,
                sub_action="s",
            ),
            StructuredAction(
                category=IDLE_ACTIVATE,
                cardspec="s1",
                card_code=44095762,
                location=LOCATION_SZONE,
                sequence=0,
                sub_action="v",
            ),
            StructuredAction(category=IDLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"], structured_actions=sa)
        obs = builder.build(gs, prompt)
        a = obs["actions"]
        m = obs["action_mask"]

        assert a.shape == (MAX_ACTIONS, ACTION_FEATURES)
        # 3 actions valid
        assert m[0] == 1
        assert m[1] == 1
        assert m[2] == 1
        assert m[3] == 0

        # Action 0: msg_type=IDLE_CMD, category=IDLE_SUMMON, code=BEWD
        assert ACTION_LAYOUT.read(a[0], "msg_type") == MSG_SELECT_IDLECMD
        assert ACTION_LAYOUT.read(a[0], "category") == IDLE_SUMMON
        assert ACTION_LAYOUT.read(a[0], "code") == 89631139

        # Action 1: msg_type=IDLE_CMD, category=IDLE_ACTIVATE
        assert ACTION_LAYOUT.read(a[1], "msg_type") == MSG_SELECT_IDLECMD
        assert ACTION_LAYOUT.read(a[1], "category") == IDLE_ACTIVATE
        assert ACTION_LAYOUT.read(a[1], "code") == 44095762

        # Action 2: end phase, category=IDLE_TO_EP
        assert ACTION_LAYOUT.read(a[2], "category") == IDLE_TO_EP

    def test_idle_index_is_per_category(self, builder, gs):
        """`index` resets per category, matching the in-process encoding."""
        sa = [
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h1",
                card_code=89631139,
                location=LOCATION_HAND,
                sequence=0,
                sub_action="s",
            ),
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h2",
                card_code=38517737,
                location=LOCATION_HAND,
                sequence=1,
                sub_action="s",
            ),
            StructuredAction(
                category=IDLE_ACTIVATE,
                cardspec="s1",
                card_code=44095762,
                location=LOCATION_SZONE,
                sequence=0,
                sub_action="v",
            ),
            StructuredAction(category=IDLE_TO_BP, sub_action="b"),
            StructuredAction(category=IDLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(
            prompt_type=PromptType.IDLE_CMD, options=["b", "e"], structured_actions=sa
        )
        obs = builder.build(gs, prompt)
        a = obs["actions"]

        # Two normal summons: index 0 and 1 within category 0
        assert ACTION_LAYOUT.read(a[0], "category") == IDLE_SUMMON
        assert ACTION_LAYOUT.read(a[0], "index") == 0
        assert ACTION_LAYOUT.read(a[1], "category") == IDLE_SUMMON  # same category
        assert ACTION_LAYOUT.read(a[1], "index") == 1
        # Activate: first in its category → index 0
        assert ACTION_LAYOUT.read(a[2], "category") == IDLE_ACTIVATE
        assert ACTION_LAYOUT.read(a[2], "index") == 0
        # Phase transitions: each is first in its category → index 0
        assert ACTION_LAYOUT.read(a[3], "category") == IDLE_TO_BP
        assert ACTION_LAYOUT.read(a[3], "index") == 0
        assert ACTION_LAYOUT.read(a[4], "category") == IDLE_TO_EP
        assert ACTION_LAYOUT.read(a[4], "index") == 0


# ---------------------------------------------------------------------------
# 8. Battle action features
# ---------------------------------------------------------------------------


class TestBattleActionFeatures:
    def test_battle_structured_actions(self, builder, gs):
        sa = [
            StructuredAction(
                category=BATTLE_ATTACK,
                cardspec="m1",
                card_code=89631139,
                location=LOCATION_MZONE,
                sequence=0,
                sub_action="m1",
            ),
            StructuredAction(category=BATTLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(
            prompt_type=PromptType.BATTLE_MENU, options=["a", "e"], structured_actions=sa
        )
        obs = builder.build(gs, prompt)
        a = obs["actions"]
        m = obs["action_mask"]

        assert ACTION_LAYOUT.read(a[0], "msg_type") == MSG_SELECT_BATTLECMD
        assert ACTION_LAYOUT.read(a[0], "category") == BATTLE_ATTACK
        assert ACTION_LAYOUT.read(a[1], "category") == BATTLE_TO_EP
        assert m[0] == 1
        assert m[1] == 1
        assert m[2] == 0


# ---------------------------------------------------------------------------
# 9. Action mask
# ---------------------------------------------------------------------------


class TestActionMask:
    def test_mask_valid_and_padding(self, builder, gs):
        sa = [StructuredAction(category=i, sub_action="e") for i in range(5)]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"], structured_actions=sa)
        obs = builder.build(gs, prompt)
        m = obs["action_mask"]

        assert m.shape == (MAX_ACTIONS,)
        assert m.dtype == np.int8
        # First 5 valid
        for i in range(5):
            assert m[i] == 1
        # Rest padding
        for i in range(5, MAX_ACTIONS):
            assert m[i] == 0

    def test_mask_max_actions_cap(self, builder, gs):
        # More actions than MAX_ACTIONS
        sa = [StructuredAction(category=IDLE_SUMMON, sub_action="e") for _ in range(40)]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"], structured_actions=sa)
        obs = builder.build(gs, prompt)
        m = obs["action_mask"]

        # Should cap at MAX_ACTIONS
        assert sum(m) == MAX_ACTIONS


# ---------------------------------------------------------------------------
# 10. Non-idle prompt actions
# ---------------------------------------------------------------------------


class TestNonIdlePromptActions:
    def test_effectyn_encodes_two_actions(self, builder, gs):
        prompt = ParsedPrompt(prompt_type=PromptType.SELECT_EFFECTYN, options=["yes", "no"])
        obs = builder.build(gs, prompt)
        m = obs["action_mask"]
        a = obs["actions"]

        assert m[0] == 1
        assert m[1] == 1
        assert m[2] == 0
        assert ACTION_LAYOUT.read(a[0], "msg_type") == MSG_SELECT_EFFECTYN
        assert ACTION_LAYOUT.read(a[1], "msg_type") == MSG_SELECT_EFFECTYN
        # RL encoding: Yes → category=0, No → category=1, both index=0
        assert ACTION_LAYOUT.read(a[0], "category") == 0  # category 0 = yes
        assert ACTION_LAYOUT.read(a[1], "category") == 1  # category 1 = no
        assert ACTION_LAYOUT.read(a[0], "index") == 0
        assert ACTION_LAYOUT.read(a[1], "index") == 0

    def test_effectyn_extracts_card_code(self, builder, tmp_db):
        from mud.card_lookup import CardNameLookup

        lookup = CardNameLookup(tmp_db)
        gs_with_lookup = MUDGameState(card_lookup=lookup)

        prompt = ParsedPrompt(
            prompt_type=PromptType.SELECT_EFFECTYN,
            options=["yes", "no"],
            raw_lines=["Do you want to use the effect from Mirror Force in s1?"],
        )
        obs = builder.build(gs_with_lookup, prompt)
        a = obs["actions"]

        # Both Yes and No should carry Mirror Force's passcode (44095762)
        code_yes = ACTION_LAYOUT.read(a[0], "code")
        code_no = ACTION_LAYOUT.read(a[1], "code")
        assert code_yes == 44095762
        assert code_no == 44095762

    def test_select_card_encodes_from_options(self, builder, gs):
        prompt = ParsedPrompt(
            prompt_type=PromptType.SELECT_CARD,
            options=["h1: Blue-Eyes", "h2: Kuriboh", "h3: Dark Magician"],
        )
        obs = builder.build(gs, prompt)
        m = obs["action_mask"]
        a = obs["actions"]

        assert m[0] == 1
        assert m[1] == 1
        assert m[2] == 1
        assert m[3] == 0
        assert ACTION_LAYOUT.read(a[0], "msg_type") == MSG_SELECT_CARD
        assert ACTION_LAYOUT.read(a[0], "index") == 0
        assert ACTION_LAYOUT.read(a[1], "index") == 1
        assert ACTION_LAYOUT.read(a[2], "index") == 2
