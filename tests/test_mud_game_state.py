"""Unit tests for the MUD game state tracker.

Pure unit tests — no cards.cdb or MUD server required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from yugioh_mud.card_lookup import CardNameLookup
from yugioh_mud.game_state import CardEntry, MUDGameState, _parse_card_line
from yugioh_mud.text_parser import EventType, ParsedEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db_path = tmp_path_factory.mktemp("cards") / "cards.cdb"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE texts (id INTEGER PRIMARY KEY, name TEXT, desc TEXT)")
    conn.execute(
        "CREATE TABLE datas (id INTEGER PRIMARY KEY, alias INTEGER)")
    conn.executemany(
        "INSERT INTO texts (id, name, desc) VALUES (?, ?, ?)",
        [
            (46986414, "Dark Magician", ""),
            (89631139, "Blue-Eyes White Dragon", ""),
            (40640057, "Kuriboh", ""),
            (44095762, "Mirror Force", ""),
        ],
    )
    conn.executemany(
        "INSERT INTO datas (id, alias) VALUES (?, ?)",
        [(46986414, 0), (89631139, 0), (40640057, 0), (44095762, 0)],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture(scope="module")
def lookup(tmp_db: Path) -> CardNameLookup:
    return CardNameLookup(tmp_db)


@pytest.fixture
def gs(lookup: CardNameLookup) -> MUDGameState:
    return MUDGameState(card_lookup=lookup)


@pytest.fixture
def gs_no_lookup() -> MUDGameState:
    return MUDGameState()


def _ev(event_type: EventType, **kwargs) -> ParsedEvent:
    return ParsedEvent(event_type=event_type, **kwargs)


# ---------------------------------------------------------------------------
# Turn / Phase
# ---------------------------------------------------------------------------

class TestTurnPhase:
    def test_new_turn_mine(self, gs: MUDGameState):
        gs.update(_ev(EventType.NEW_TURN, player="you"))
        assert gs.turn == 1
        assert gs.is_my_turn is True

    def test_new_turn_opponent(self, gs: MUDGameState):
        gs.update(_ev(EventType.NEW_TURN, player="Player2", is_opponent=True))
        assert gs.turn == 1
        assert gs.is_my_turn is False

    def test_consecutive_turns(self, gs: MUDGameState):
        gs.update(_ev(EventType.NEW_TURN, player="you"))
        gs.update(_ev(EventType.NEW_TURN, player="Player2", is_opponent=True))
        gs.update(_ev(EventType.NEW_TURN, player="you"))
        assert gs.turn == 3
        assert gs.is_my_turn is True

    def test_phase(self, gs: MUDGameState):
        gs.update(_ev(EventType.NEW_PHASE, phase="main1 phase"))
        assert gs.phase == "main1 phase"
        gs.update(_ev(EventType.NEW_PHASE, phase="battle phase"))
        assert gs.phase == "battle phase"


# ---------------------------------------------------------------------------
# LP
# ---------------------------------------------------------------------------

class TestLP:
    def test_my_damage(self, gs: MUDGameState):
        gs.update(_ev(EventType.DAMAGE, player="you", amount=1500, new_lp=6500))
        assert gs.my_lp == 6500

    def test_opp_damage(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.DAMAGE, player="P2", is_opponent=True,
            amount=2000, new_lp=6000))
        assert gs.opp_lp == 6000

    def test_my_recover(self, gs: MUDGameState):
        gs.update(_ev(EventType.DAMAGE, player="you", amount=3000, new_lp=5000))
        gs.update(_ev(EventType.RECOVER, player="you", amount=500, new_lp=5500))
        assert gs.my_lp == 5500

    def test_opp_recover(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.RECOVER, player="P2", is_opponent=True,
            amount=1000, new_lp=9000))
        assert gs.opp_lp == 9000

    def test_my_pay_lp(self, gs: MUDGameState):
        gs.update(_ev(EventType.PAY_LP, player="you", amount=1000, new_lp=7000))
        assert gs.my_lp == 7000

    def test_opp_pay_lp(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.PAY_LP, player="P2", is_opponent=True,
            amount=800, new_lp=7200))
        assert gs.opp_lp == 7200


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

class TestDraw:
    def test_my_draw_header_does_not_add_blanks(self, gs: MUDGameState):
        """DRAW event alone no longer adds blank entries — DRAW_CARD does."""
        gs.update(_ev(EventType.DRAW, player="you", amount=5))
        assert len(gs.my_hand) == 0

    def test_my_draw_card_adds_named_entry(self, gs: MUDGameState):
        gs.update(_ev(EventType.DRAW, player="you", amount=2))
        gs.update(_ev(EventType.DRAW_CARD, player="you",
                       card_name="Dark Magician"))
        gs.update(_ev(EventType.DRAW_CARD, player="you",
                       card_name="Blue-Eyes White Dragon"))
        assert len(gs.my_hand) == 2
        assert gs.my_hand[0].name == "Dark Magician"
        assert gs.my_hand[0].code == 46986414
        assert gs.my_hand[1].name == "Blue-Eyes White Dragon"
        assert gs.my_hand[1].code == 89631139

    def test_opp_draw(self, gs: MUDGameState):
        gs.update(_ev(EventType.DRAW, player="P2", is_opponent=True, amount=5))
        assert gs.opp_hand_count == 5

    def test_multiple_draws(self, gs: MUDGameState):
        gs.update(_ev(EventType.DRAW, player="you", amount=3))
        for name in ("Dark Magician", "Blue-Eyes White Dragon", "Red-Eyes Black Dragon"):
            gs.update(_ev(EventType.DRAW_CARD, player="you", card_name=name))
        gs.update(_ev(EventType.DRAW, player="you", amount=1))
        gs.update(_ev(EventType.DRAW_CARD, player="you",
                       card_name="Dark Magician"))
        assert len(gs.my_hand) == 4


# ---------------------------------------------------------------------------
# Summon / Set
# ---------------------------------------------------------------------------

class TestSummon:
    def test_normal_summon(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SUMMON, player="Player1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Dark Magician"
        assert gs.my_mzone[0].code == 46986414
        assert gs.my_mzone[0].position == "face-up attack"

    def test_opp_summon(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SUMMON, player="P2", is_opponent=True,
            card_name="Blue-Eyes White Dragon", position="face-up attack"))
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].name == "Blue-Eyes White Dragon"
        assert gs.opp_mzone[0].code == 89631139

    def test_special_summon(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SP_SUMMON, player="Player1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs.my_mzone) == 1

    def test_flip_summon(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.FLIP_SUMMON, player="Player1",
            card_name="Kuriboh", card_spec="m1"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].spec == "m1"

    def test_flip_summon_in_place(self, gs: MUDGameState):
        """Flip summon updates existing face-down card instead of appending."""
        # Set a face-down monster first
        gs.update(_ev(
            EventType.SET, player="you", card_spec="m1",
            card_name="", position="face-down defense"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].position == "face-down defense"
        assert gs.my_mzone[0].name == ""
        # Flip summon reveals it — should update in place, not duplicate
        gs.update(_ev(
            EventType.FLIP_SUMMON, player="Player1",
            card_name="Kuriboh", card_spec="m1"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Kuriboh"
        assert gs.my_mzone[0].code == 40640057
        assert gs.my_mzone[0].position == "face-up attack"
        assert gs.my_mzone[0].spec == "m1"

    def test_set_monster(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SET, player="you", card_spec="m1",
            card_name="Kuriboh", position="face-down defense"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].position == "face-down defense"

    def test_set_spell(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SET, player="you", card_spec="s1",
            card_name="Mirror Force", position="face-down"))
        assert len(gs.my_szone) == 1
        assert gs.my_szone[0].name == "Mirror Force"

    def test_opp_set(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SET, player="P2", is_opponent=True,
            card_spec="m1", position="face-down defense"))
        assert len(gs.opp_mzone) == 1
        # Opponent set — no card name visible
        assert gs.opp_mzone[0].name == ""

    def test_opp_set_spell_with_o_prefix(self, gs: MUDGameState):
        """Opponent set with 'os1' spec routes to opp_szone, not opp_mzone."""
        gs.update(_ev(
            EventType.SET, player="P2", is_opponent=True,
            card_spec="os1", position="face-down"))
        assert len(gs.opp_szone) == 1
        assert len(gs.opp_mzone) == 0
        assert gs.opp_szone[0].spec == "os1"

    def test_opp_set_monster_with_o_prefix(self, gs: MUDGameState):
        """Opponent set with 'om1' spec routes to opp_mzone, not opp_szone."""
        gs.update(_ev(
            EventType.SET, player="P2", is_opponent=True,
            card_spec="om1", position="face-down defense"))
        assert len(gs.opp_mzone) == 1
        assert len(gs.opp_szone) == 0
        assert gs.opp_mzone[0].spec == "om1"

    # -- Fix 1: own summon/set removes from hand --

    def test_summon_removes_from_hand(self, gs: MUDGameState):
        """Normal summon should remove the card from own hand."""
        gs.my_hand.append(CardEntry(
            name="Dark Magician", code=46986414, spec="h1"))
        gs.update(_ev(
            EventType.SUMMON, player="P1",
            card_name="Dark Magician", card_spec="m1",
            position="face-up attack"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_mzone) == 1

    def test_set_removes_from_hand(self, gs: MUDGameState):
        """Setting a monster should remove from own hand."""
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.SET, player="you", card_spec="m1",
            card_name="Kuriboh", position="face-down defense"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_mzone) == 1

    def test_set_spell_removes_from_hand(self, gs: MUDGameState):
        """Setting a spell/trap should remove from own hand."""
        gs.my_hand.append(CardEntry(
            name="Mirror Force", code=44095762, spec="h1"))
        gs.update(_ev(
            EventType.SET, player="you", card_spec="s1",
            card_name="Mirror Force", position="face-down"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_szone) == 1

    def test_sp_summon_removes_from_hand(self, gs: MUDGameState):
        """Special summon should speculatively remove from hand."""
        gs.my_hand.append(CardEntry(
            name="Dark Magician", code=46986414, spec="h1"))
        gs.update(_ev(
            EventType.SP_SUMMON, player="P1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_mzone) == 1

    def test_sp_summon_from_gy_doesnt_remove_hand(self, gs: MUDGameState):
        """SP summon from GY: hand card with same name should NOT be removed
        if card isn't actually in hand (speculative removal is a no-op)."""
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414))
        # No Dark Magician in hand
        gs.update(_ev(
            EventType.SP_SUMMON, player="P1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_mzone) == 1

    # -- Fix 2: opponent summon/set decrements opp_hand_count --

    def test_opp_summon_decrements_hand(self, gs: MUDGameState):
        """Opponent normal summon should decrement opp_hand_count."""
        gs.opp_hand_count = 5
        gs.update(_ev(
            EventType.SUMMON, player="P2", is_opponent=True,
            card_name="Blue-Eyes White Dragon", position="face-up attack"))
        assert gs.opp_hand_count == 4
        assert len(gs.opp_mzone) == 1

    def test_opp_set_decrements_hand(self, gs: MUDGameState):
        """Opponent set should decrement opp_hand_count."""
        gs.opp_hand_count = 3
        gs.update(_ev(
            EventType.SET, player="P2", is_opponent=True,
            card_spec="om1", position="face-down defense"))
        assert gs.opp_hand_count == 2

    def test_opp_sp_summon_decrements_hand(self, gs: MUDGameState):
        """Opponent special summon should decrement opp_hand_count."""
        gs.opp_hand_count = 4
        gs.update(_ev(
            EventType.SP_SUMMON, player="P2", is_opponent=True,
            card_name="Blue-Eyes White Dragon", position="face-up attack"))
        assert gs.opp_hand_count == 3

    def test_opp_hand_count_never_negative(self, gs: MUDGameState):
        """opp_hand_count should never go below 0."""
        gs.opp_hand_count = 0
        gs.update(_ev(
            EventType.SUMMON, player="P2", is_opponent=True,
            card_name="Blue-Eyes White Dragon", position="face-up attack"))
        assert gs.opp_hand_count == 0


# ---------------------------------------------------------------------------
# Position change
# ---------------------------------------------------------------------------

class TestPosChange:
    def test_pos_change(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SUMMON, player="P1",
            card_name="Dark Magician", card_spec="m1",
            position="face-up attack"))
        gs.update(_ev(
            EventType.POS_CHANGE, card_spec="m1",
            card_name="Dark Magician", position="face-up defense"))
        assert gs.my_mzone[0].position == "face-up defense"


# ---------------------------------------------------------------------------
# Card movement
# ---------------------------------------------------------------------------

class TestMovement:
    def _summon(self, gs: MUDGameState, name: str, spec: str,
                opponent: bool = False) -> None:
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "P1",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))
        # Set spec on the entry
        zone = gs.opp_mzone if opponent else gs.my_mzone
        zone[-1].spec = spec

    def test_destroy(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        assert len(gs.my_mzone) == 1
        gs.update(_ev(
            EventType.DESTROY, card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0

    def test_to_graveyard(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Dark Magician"

    def test_opp_to_graveyard(self, gs: MUDGameState):
        self._summon(gs, "Blue-Eyes White Dragon", "m1", opponent=True)
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="P2", is_opponent=True,
            card_spec="m1", card_name="Blue-Eyes White Dragon"))
        assert len(gs.opp_mzone) == 0
        assert len(gs.opp_graveyard) == 1

    def test_banished(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.BANISHED, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0
        assert len(gs.my_banished) == 1

    def test_to_hand(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.TO_HAND, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0
        assert any(c.name == "Dark Magician" for c in gs.my_hand)

    def test_opp_to_hand(self, gs: MUDGameState):
        self._summon(gs, "Blue-Eyes White Dragon", "m1", opponent=True)
        gs.update(_ev(
            EventType.TO_HAND, player="P2", is_opponent=True,
            card_spec="m1", card_name="Blue-Eyes White Dragon"))
        assert len(gs.opp_mzone) == 0
        assert gs.opp_hand_count == 1

    def test_to_deck(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.TO_DECK, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0

    def test_tribute(self, gs: MUDGameState):
        self._summon(gs, "Kuriboh", "m1")
        gs.update(_ev(
            EventType.TRIBUTE, player="you",
            card_spec="m1", card_name="Kuriboh"))
        assert len(gs.my_mzone) == 0

    def test_discard_from_hand(self, gs: MUDGameState):
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.DISCARD, player="you",
            card_spec="h1", card_name="Kuriboh"))
        assert len(gs.my_hand) == 0

    def test_opp_discard(self, gs: MUDGameState):
        gs.opp_hand_count = 5
        gs.update(_ev(
            EventType.DISCARD, player="P2", is_opponent=True,
            card_spec="h1", card_name="Kuriboh"))
        assert gs.opp_hand_count == 4

    # -- Fix 3: chaining from hand --

    def test_chaining_from_own_hand(self, gs: MUDGameState):
        """Activating a hand trap (chaining from h-spec) removes from hand."""
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.CHAINING, player="you",
            card_spec="h1", card_name="Kuriboh"))
        assert len(gs.my_hand) == 0

    def test_chaining_from_opp_hand(self, gs: MUDGameState):
        """Opponent activating from hand decrements opp_hand_count."""
        gs.opp_hand_count = 4
        gs.update(_ev(
            EventType.CHAINING, player="P2", is_opponent=True,
            card_spec="oh3", card_name="Effect Veiler"))
        assert gs.opp_hand_count == 3

    def test_chaining_from_field_no_hand_removal(self, gs: MUDGameState):
        """Chaining from field spec should NOT remove from hand."""
        gs.my_hand.append(CardEntry(
            name="Mirror Force", code=44095762, spec="h1"))
        gs.update(_ev(
            EventType.CHAINING, player="you",
            card_spec="s1", card_name="Mirror Force"))
        # Hand should be untouched — chaining from field
        assert len(gs.my_hand) == 1

    # -- Fix 5: TO_GRAVEYARD from hand --

    def test_to_graveyard_from_own_hand(self, gs: MUDGameState):
        """Card sent from hand to GY should remove from hand and add to GY."""
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="h1", card_name="Kuriboh"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Kuriboh"

    def test_to_graveyard_from_opp_hand(self, gs: MUDGameState):
        """Opponent card sent from hand to GY should decrement opp_hand_count."""
        gs.opp_hand_count = 3
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="P2", is_opponent=True,
            card_spec="oh2", card_name="Effect Veiler"))
        assert gs.opp_hand_count == 2
        assert len(gs.opp_graveyard) == 1

    def test_banished_from_own_hand(self, gs: MUDGameState):
        """Card banished from hand should remove from hand."""
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.BANISHED, player="you",
            card_spec="h1", card_name="Kuriboh"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_banished) == 1
        assert gs.my_banished[0].name == "Kuriboh"

    def test_banished_from_opp_hand(self, gs: MUDGameState):
        """Opponent card banished from hand should decrement opp_hand_count."""
        gs.opp_hand_count = 2
        gs.update(_ev(
            EventType.BANISHED, player="P2", is_opponent=True,
            card_spec="oh1", card_name="Effect Veiler"))
        assert gs.opp_hand_count == 1
        assert len(gs.opp_banished) == 1


# ---------------------------------------------------------------------------
# No-lookup mode
# ---------------------------------------------------------------------------

class TestNoLookup:
    def test_summon_without_lookup(self, gs_no_lookup: MUDGameState):
        gs_no_lookup.update(_ev(
            EventType.SUMMON, player="P1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs_no_lookup.my_mzone) == 1
        assert gs_no_lookup.my_mzone[0].code == 0
        assert gs_no_lookup.my_mzone[0].name == "Dark Magician"


# ---------------------------------------------------------------------------
# Resync — score
# ---------------------------------------------------------------------------

class TestResyncScore:
    def test_resync_score_lp(self, gs: MUDGameState):
        gs.my_lp = 7500  # drifted
        gs.resync_score([
            "Your LP: 8000 Opponent LP: 6000",
            "Hand: You: 5 Opponent: 4",
            "Deck: You: 35 Opponent: 36",
            "Grave: You: 0 Opponent: 0",
            "Removed: You: 0 Opponent: 0",
            "It's your turn.",
        ])
        assert gs.my_lp == 8000
        assert gs.opp_lp == 6000
        assert gs.is_my_turn is True

    def test_resync_score_opp_turn(self, gs: MUDGameState):
        gs.resync_score(["It's Player2's turn."])
        assert gs.is_my_turn is False

    def test_resync_score_hand_count(self, gs: MUDGameState):
        gs.opp_hand_count = 3  # drifted
        gs.resync_score(["Hand: You: 5 Opponent: 5"])
        assert gs.opp_hand_count == 5


# ---------------------------------------------------------------------------
# Resync — hand
# ---------------------------------------------------------------------------

class TestResyncHand:
    def test_resync_hand(self, gs: MUDGameState):
        gs.my_hand = [CardEntry(name="stale")]
        gs.resync_hand([
            "h1 Dark Magician",
            "h2 Kuriboh",
        ])
        assert len(gs.my_hand) == 2
        assert gs.my_hand[0].name == "Dark Magician"
        assert gs.my_hand[0].code == 46986414
        assert gs.my_hand[1].name == "Kuriboh"

    def test_resync_hand_empty(self, gs: MUDGameState):
        gs.my_hand = [CardEntry(name="stale")]
        gs.resync_hand(["No cards."])
        assert len(gs.my_hand) == 0


# ---------------------------------------------------------------------------
# Resync — tab
# ---------------------------------------------------------------------------

class TestResyncTab:
    def test_resync_tab_own(self, gs: MUDGameState):
        gs.resync_tab([
            "Your table:",
            "m1: Dark Magician (2500/2100) level 7 face-up attack",
            "s1: Mirror Force face-down",
        ])
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Dark Magician"
        assert gs.my_mzone[0].code == 46986414
        assert len(gs.my_szone) == 1
        assert gs.my_szone[0].name == "Mirror Force"

    def test_resync_tab_opponent(self, gs: MUDGameState):
        gs.resync_tab([
            "Opponent's table:",
            "m1: face-down defense",
        ], opponent=True)
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].name == ""
        assert gs.opp_mzone[0].position == "face-down defense"

    def test_resync_tab_empty(self, gs: MUDGameState):
        gs.my_mzone = [CardEntry(name="stale")]
        gs.resync_tab([
            "Your table:",
            "Table is empty.",
        ])
        assert len(gs.my_mzone) == 0
        assert len(gs.my_szone) == 0

    def test_resync_tab_szone_face_down_space(self, gs: MUDGameState):
        """MUD server sends 'face down' (space) not 'face-down' (hyphen)."""
        gs.resync_tab([
            "Your table:",
            "s1: face down",
        ])
        assert len(gs.my_szone) == 1
        assert gs.my_szone[0].name == ""
        assert gs.my_szone[0].position == "face down"

    def test_resync_tab_mzone_face_down_space(self, gs: MUDGameState):
        """Monster zone face-down with space separator."""
        gs.resync_tab([
            "Your table:",
            "m1: face down defense",
        ])
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == ""
        assert "face" in gs.my_mzone[0].position

    def test_resync_tab_szone_face_hyphen_still_works(self, gs: MUDGameState):
        """Hyphenated 'face-down' format should still be matched."""
        gs.resync_tab([
            "Your table:",
            "s1: face-down",
        ])
        assert len(gs.my_szone) == 1
        assert gs.my_szone[0].position == "face-down"


# ---------------------------------------------------------------------------
# Chaining — opponent bare spec (P0 Fix 3)
# ---------------------------------------------------------------------------

class TestChainingBareSpec:
    def test_opp_chain_bare_spec_decrements_hand(self, gs: MUDGameState):
        """Opponent chains a spell from hand with bare card name (no zone prefix)."""
        gs.opp_hand_count = 5
        gs.update(_ev(
            EventType.CHAINING, player="P2", is_opponent=True,
            card_spec="Mystical Space Typhoon",
            card_name="Mystical Space Typhoon"))
        assert gs.opp_hand_count == 4

    def test_opp_chain_empty_spec_decrements_hand(self, gs: MUDGameState):
        """Opponent chains with empty spec (from hand)."""
        gs.opp_hand_count = 3
        gs.update(_ev(
            EventType.CHAINING, player="P2", is_opponent=True,
            card_spec="", card_name="Effect Veiler"))
        assert gs.opp_hand_count == 2

    def test_opp_chain_field_spec_no_hand_decrement(self, gs: MUDGameState):
        """Opponent chains from field (os/om prefix) should NOT decrement hand."""
        gs.opp_hand_count = 3
        gs.opp_szone.append(CardEntry(
            name="Mirror Force", code=44095762, spec="os1"))
        gs.update(_ev(
            EventType.CHAINING, player="P2", is_opponent=True,
            card_spec="os1", card_name="Mirror Force"))
        assert gs.opp_hand_count == 3


# ---------------------------------------------------------------------------
# P1: Tribute / Destroy / Discard add to GY
# ---------------------------------------------------------------------------

class TestTributeAddsToGY:
    def _summon(self, gs, name, spec, opponent=False):
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "you",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))

    def test_tribute_own_adds_to_gy(self, gs: MUDGameState):
        self._summon(gs, "Kuriboh", "m1")
        gs.update(_ev(
            EventType.TRIBUTE, player="you",
            card_spec="m1", card_name="Kuriboh"))
        assert len(gs.my_mzone) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Kuriboh"

    def test_tribute_opp_adds_to_gy(self, gs: MUDGameState):
        self._summon(gs, "Kuriboh", "m1", opponent=True)
        gs.update(_ev(
            EventType.TRIBUTE, player="P2", is_opponent=True,
            card_spec="m1", card_name="Kuriboh"))
        assert len(gs.opp_mzone) == 0
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == "Kuriboh"


class TestDestroyAddsToGY:
    def _summon(self, gs, name, spec, opponent=False):
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "you",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))

    def test_destroy_own_adds_to_gy(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.DESTROY, card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_mzone) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Dark Magician"

    def test_destroy_opp_adds_to_gy(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1", opponent=True)
        gs.update(_ev(
            EventType.DESTROY, is_opponent=True,
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.opp_mzone) == 0
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == "Dark Magician"


class TestDiscardAddsToGY:
    def test_discard_own_adds_to_gy(self, gs: MUDGameState):
        gs.my_hand.append(CardEntry(
            name="Kuriboh", code=40640057, spec="h1"))
        gs.update(_ev(
            EventType.DISCARD, player="you",
            card_spec="h1", card_name="Kuriboh"))
        assert len(gs.my_hand) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Kuriboh"

    def test_discard_opp_adds_to_gy(self, gs: MUDGameState):
        gs.opp_hand_count = 3
        gs.update(_ev(
            EventType.DISCARD, player="P2", is_opponent=True,
            card_spec="h1", card_name="Kuriboh"))
        assert gs.opp_hand_count == 2
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == "Kuriboh"


# ---------------------------------------------------------------------------
# P1: Mill spec guard (deck → GY without false removal)
# ---------------------------------------------------------------------------

class TestMillSpecGuard:
    def test_mill_does_not_remove_existing_gy_card(self, gs: MUDGameState):
        """Milling a card from deck should not remove a same-named card from GY."""
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="", card_name="Dark Magician"))
        # Should have 2 copies now (existing + milled), not 1
        assert len(gs.my_graveyard) == 2

    def test_mill_opp_does_not_remove_existing(self, gs: MUDGameState):
        """Opponent mill should not remove existing same-named card."""
        gs.opp_graveyard.append(CardEntry(
            name="Blue-Eyes White Dragon", code=89631139))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="P2", is_opponent=True,
            card_spec="", card_name="Blue-Eyes White Dragon"))
        assert len(gs.opp_graveyard) == 2

    def test_banish_from_deck_does_not_remove_existing(self, gs: MUDGameState):
        """Banishing from deck should not remove existing same-named card."""
        gs.my_banished.append(CardEntry(
            name="Dark Magician", code=46986414))
        gs.update(_ev(
            EventType.BANISHED, player="you",
            card_spec="", card_name="Dark Magician"))
        assert len(gs.my_banished) == 2


# ---------------------------------------------------------------------------
# P2: Spec prefix normalization
# ---------------------------------------------------------------------------

class TestSpecPrefixNorm:
    def _summon(self, gs, name, spec, opponent=False):
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "you",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))

    def test_remove_from_field_oprefix_match(self, gs: MUDGameState):
        """Card stored as 'os1' should be found by spec 's1' and vice-versa."""
        gs.opp_szone.append(CardEntry(
            name="Mirror Force", code=44095762, spec="os1"))
        removed = gs._remove_from_field("s1", "Mirror Force")
        assert removed is not None
        assert removed.name == "Mirror Force"
        assert len(gs.opp_szone) == 0

    def test_remove_from_field_no_prefix_finds_oprefix(self, gs: MUDGameState):
        """Card stored as 'm1' should be found by spec 'om1'."""
        gs.opp_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        removed = gs._remove_from_field("om1", "Dark Magician")
        assert removed is not None
        assert len(gs.opp_mzone) == 0

    def test_exact_match_still_works(self, gs: MUDGameState):
        gs.my_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        removed = gs._remove_from_field("m1", "Dark Magician")
        assert removed is not None
        assert len(gs.my_mzone) == 0


# ---------------------------------------------------------------------------
# P2: FROM_GY / SP_SUMMON dedup
# ---------------------------------------------------------------------------

class TestFromGYDedup:
    def test_from_gy_then_sp_summon_no_double_add(self, gs: MUDGameState):
        """FROM_GY_TO_FIELD + SP_SUMMON for same card should not double-add."""
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414))
        # FROM_GY adds to field
        gs.update(_ev(
            EventType.FROM_GY_TO_FIELD, player="you",
            card_spec="g1", card_name="Dark Magician",
            target_spec="m1"))
        assert len(gs.my_mzone) == 1
        # SP_SUMMON fires immediately after for same card
        gs.update(_ev(
            EventType.SP_SUMMON, player="you",
            card_name="Dark Magician", card_spec="m1",
            position="face-up attack"))
        # Should still be 1, not 2
        assert len(gs.my_mzone) == 1

    def test_from_gy_then_sp_summon_opp(self, gs: MUDGameState):
        gs.opp_graveyard.append(CardEntry(
            name="Blue-Eyes White Dragon", code=89631139))
        gs.update(_ev(
            EventType.FROM_GY_TO_FIELD, player="P2", is_opponent=True,
            card_spec="og1", card_name="Blue-Eyes White Dragon",
            target_spec="om4"))
        assert len(gs.opp_mzone) == 1
        gs.update(_ev(
            EventType.SP_SUMMON, player="P2", is_opponent=True,
            card_name="Blue-Eyes White Dragon", card_spec="om4",
            position="face-up attack"))
        assert len(gs.opp_mzone) == 1


# ---------------------------------------------------------------------------
# P2: Cross-player DESTROY → correct GY
# ---------------------------------------------------------------------------

class TestCrossPlayerDestroy:
    def test_destroy_opp_card_goes_to_opp_gy(self, gs: MUDGameState):
        """DESTROY with 'os1' spec should add to opp GY, not own."""
        gs.opp_szone.append(CardEntry(
            name="One for One", code=2295440, spec="os1"))
        gs.update(_ev(
            EventType.DESTROY, card_spec="os1", card_name="One for One"))
        assert len(gs.opp_szone) == 0
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == "One for One"
        assert len(gs.my_graveyard) == 0

    def test_destroy_own_card_goes_to_own_gy(self, gs: MUDGameState):
        """DESTROY with 's1' (no o prefix) should add to own GY."""
        gs.my_szone.append(CardEntry(
            name="Mirror Force", code=44095762, spec="s1"))
        gs.update(_ev(
            EventType.DESTROY, card_spec="s1", card_name="Mirror Force"))
        assert len(gs.my_szone) == 0
        assert len(gs.my_graveyard) == 1
        assert len(gs.opp_graveyard) == 0


# ---------------------------------------------------------------------------
# P2: Spell activation from hand → GY (dedup guard refinement)
# ---------------------------------------------------------------------------

class TestSpellActivationFromHandToGY:
    def test_spell_from_hand_then_to_gy(self, gs: MUDGameState):
        """Spell activated from hand then sent to GY should land in GY."""
        gs.my_hand.append(CardEntry(
            name="Raigeki", code=12580477, spec="h1"))
        # CHAINING removes from hand
        gs.update(_ev(
            EventType.CHAINING, player="you",
            card_spec="h1", card_name="Raigeki"))
        assert len(gs.my_hand) == 0
        # TO_GRAVEYARD with field spec "s3" — card was never on szone
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="s3", card_name="Raigeki"))
        # Should be in GY (not skipped by dedup guard)
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Raigeki"

    def test_destroy_then_to_gy_still_deduped(self, gs: MUDGameState):
        """DESTROY → TO_GRAVEYARD for same card should not double-add."""
        gs.my_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        # DESTROY adds to GY
        gs.update(_ev(
            EventType.DESTROY, card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_graveyard) == 1
        # TO_GRAVEYARD fires after — should be deduped
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert len(gs.my_graveyard) == 1  # Still 1, not 2


# ---------------------------------------------------------------------------
# Extra deck tracking
# ---------------------------------------------------------------------------

class TestExtraDeck:
    def _summon(self, gs, name, spec, opponent=False):
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "P1",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))
        zone = gs.opp_mzone if opponent else gs.my_mzone
        zone[-1].spec = spec

    def test_to_extra_deck_own(self, gs: MUDGameState):
        self._summon(gs, "Decode Talker", "m1")
        gs.update(_ev(
            EventType.TO_EXTRA_DECK, player="you",
            card_spec="m1", card_name="Decode Talker"))
        assert len(gs.my_mzone) == 0
        assert len(gs.my_extra) == 1
        assert gs.my_extra[0].name == "Decode Talker"

    def test_to_extra_deck_opp(self, gs: MUDGameState):
        self._summon(gs, "Decode Talker", "m1", opponent=True)
        gs.update(_ev(
            EventType.TO_EXTRA_DECK, player="P2", is_opponent=True,
            card_spec="m1", card_name="Decode Talker"))
        assert len(gs.opp_mzone) == 0
        assert len(gs.opp_extra) == 1
        # Opponent extra — name hidden
        assert gs.opp_extra[0].name == ""


# ---------------------------------------------------------------------------
# GY/Banished → Field
# ---------------------------------------------------------------------------

class TestFromGYBanished:
    def test_from_gy_to_field_own(self, gs: MUDGameState):
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414, spec="g1"))
        gs.update(_ev(
            EventType.FROM_GY_TO_FIELD, player="you",
            card_spec="g1", card_name="Dark Magician", target_spec="m2"))
        assert len(gs.my_graveyard) == 0
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Dark Magician"
        assert gs.my_mzone[0].spec == "m2"

    def test_from_gy_to_field_opp(self, gs: MUDGameState):
        gs.opp_graveyard.append(CardEntry(
            name="Kuriboh", code=40640057, spec="og1"))
        gs.update(_ev(
            EventType.FROM_GY_TO_FIELD, player="P2", is_opponent=True,
            card_spec="og1", card_name="Kuriboh", target_spec="om1"))
        assert len(gs.opp_graveyard) == 0
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].name == "Kuriboh"

    def test_from_gy_to_szone(self, gs: MUDGameState):
        gs.my_graveyard.append(CardEntry(
            name="Mirror Force", code=44095762, spec="g1"))
        gs.update(_ev(
            EventType.FROM_GY_TO_FIELD, player="you",
            card_spec="g1", card_name="Mirror Force", target_spec="s3"))
        assert len(gs.my_graveyard) == 0
        assert len(gs.my_szone) == 1

    def test_from_banished_to_field_own(self, gs: MUDGameState):
        gs.my_banished.append(CardEntry(
            name="Dark Magician", code=46986414, spec="r1"))
        gs.update(_ev(
            EventType.FROM_BANISHED_TO_FIELD, player="you",
            card_spec="r1", card_name="Dark Magician", target_spec="m3"))
        assert len(gs.my_banished) == 0
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Dark Magician"

    def test_from_banished_to_field_opp(self, gs: MUDGameState):
        gs.opp_banished.append(CardEntry(
            name="Kuriboh", code=40640057, spec="or1"))
        gs.update(_ev(
            EventType.FROM_BANISHED_TO_FIELD, player="P2", is_opponent=True,
            card_spec="or1", card_name="Kuriboh", target_spec="om2"))
        assert len(gs.opp_banished) == 0
        assert len(gs.opp_mzone) == 1


# ---------------------------------------------------------------------------
# Non-field zone removal (GY→hand, banished→GY, etc.)
# ---------------------------------------------------------------------------

class TestNonfieldRemoval:
    def test_gy_to_hand(self, gs: MUDGameState):
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414, spec="g1"))
        gs.update(_ev(
            EventType.TO_HAND, player="you",
            card_spec="g1", card_name="Dark Magician"))
        assert len(gs.my_graveyard) == 0
        assert any(c.name == "Dark Magician" for c in gs.my_hand)

    def test_banished_to_gy(self, gs: MUDGameState):
        gs.my_banished.append(CardEntry(
            name="Kuriboh", code=40640057, spec="r1"))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="r1", card_name="Kuriboh"))
        assert len(gs.my_banished) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Kuriboh"

    def test_gy_to_banished(self, gs: MUDGameState):
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414, spec="g1"))
        gs.update(_ev(
            EventType.BANISHED, player="you",
            card_spec="g1", card_name="Dark Magician"))
        assert len(gs.my_graveyard) == 0
        assert len(gs.my_banished) == 1

    def test_banished_to_deck(self, gs: MUDGameState):
        gs.my_banished.append(CardEntry(
            name="Dark Magician", code=46986414, spec="r1"))
        gs.update(_ev(
            EventType.TO_DECK, player="you",
            card_spec="r1", card_name="Dark Magician"))
        assert len(gs.my_banished) == 0

    def test_extra_to_gy(self, gs: MUDGameState):
        gs.my_extra.append(CardEntry(
            name="Decode Talker", spec="x1"))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="x1", card_name="Decode Talker"))
        assert len(gs.my_extra) == 0
        assert len(gs.my_graveyard) == 1


# ---------------------------------------------------------------------------
# Control change / Zone switch / Swap
# ---------------------------------------------------------------------------

class TestControlChange:
    def test_your_card_changes_controller(self, gs: MUDGameState):
        """Our card on m1 moves to opponent's field at om2."""
        gs.my_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        gs.update(_ev(
            EventType.CONTROL_CHANGE, player="you",
            card_spec="m1", card_name="Dark Magician",
            target_spec="om2"))
        assert len(gs.my_mzone) == 0
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].name == "Dark Magician"
        assert gs.opp_mzone[0].spec == "om2"

    def test_opp_card_changes_to_us(self, gs: MUDGameState):
        """Opponent's card comes to our field."""
        gs.opp_mzone.append(CardEntry(
            name="Blue-Eyes White Dragon", code=89631139, spec="om1"))
        gs.update(_ev(
            EventType.CONTROL_CHANGE, player="P2", is_opponent=True,
            card_spec="om1", card_name="Blue-Eyes White Dragon",
            target_spec="m3"))
        assert len(gs.opp_mzone) == 0
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].spec == "m3"


class TestZoneSwitch:
    def test_my_card_switches_zone(self, gs: MUDGameState):
        gs.my_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        gs.update(_ev(
            EventType.ZONE_SWITCH, player="you",
            card_spec="m1", card_name="Dark Magician",
            target_spec="m3"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].spec == "m3"

    def test_opp_card_switches_zone(self, gs: MUDGameState):
        gs.opp_mzone.append(CardEntry(
            name="Blue-Eyes White Dragon", code=89631139, spec="om1"))
        gs.update(_ev(
            EventType.ZONE_SWITCH, player="P2", is_opponent=True,
            card_spec="om1", card_name="Blue-Eyes White Dragon",
            target_spec="om3"))
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].spec == "om3"


class TestSwap:
    def test_swap_moves_card(self, gs: MUDGameState):
        gs.my_mzone.append(CardEntry(
            name="Dark Magician", code=46986414, spec="m1"))
        gs.update(_ev(
            EventType.SWAP, card_name="Dark Magician",
            target_spec="om2"))
        assert len(gs.my_mzone) == 0
        assert len(gs.opp_mzone) == 1
        assert gs.opp_mzone[0].spec == "om2"


# ---------------------------------------------------------------------------
# _parse_card_line
# ---------------------------------------------------------------------------

class TestParseCardLine:
    def test_basic(self):
        name, pos = _parse_card_line("Dark Magician face-up attack")
        assert name == "Dark Magician"
        assert pos == "face-up attack"

    def test_with_level(self):
        name, pos = _parse_card_line(
            "Dark Magician face-up attack level 7")
        assert name == "Dark Magician"
        assert pos == "face-up attack"

    def test_tricky_name(self):
        name, pos = _parse_card_line(
            "Attack Gainer face-up attack level 4")
        assert name == "Attack Gainer"
        assert pos == "face-up attack"

    def test_facedown(self):
        name, pos = _parse_card_line("face-down defense")
        assert name == ""
        assert pos == "face-down defense"

    def test_link_rating(self):
        name, pos = _parse_card_line(
            "Decode Talker face-up attack link rating 3")
        assert name == "Decode Talker"
        assert pos == "face-up attack"

    def test_rank(self):
        name, pos = _parse_card_line(
            "Number 39: Utopia face-up attack rank 4")
        assert name == "Number 39: Utopia"
        assert pos == "face-up attack"

    def test_face_down_no_level(self):
        name, pos = _parse_card_line("face down")
        assert name == ""
        assert pos == "face down"

    def test_level_warrior(self):
        """Card named 'Level Warrior' should not be confused with level suffix."""
        name, pos = _parse_card_line(
            "Level Warrior face-up attack level 3")
        assert name == "Level Warrior"
        assert pos == "face-up attack"


# ---------------------------------------------------------------------------
# Resync — grave
# ---------------------------------------------------------------------------

class TestResyncGrave:
    def test_resync_grave_own(self, gs: MUDGameState):
        gs.resync_grave([
            "g1 Dark Magician face-up attack level 7",
            "g2 Kuriboh face-up defense level 1",
        ])
        assert len(gs.my_graveyard) == 2
        assert gs.my_graveyard[0].name == "Dark Magician"
        assert gs.my_graveyard[0].spec == "g1"
        assert gs.my_graveyard[0].code == 46986414
        assert gs.my_graveyard[1].name == "Kuriboh"

    def test_resync_grave_opp(self, gs: MUDGameState):
        gs.resync_grave([
            "og1 Blue-Eyes White Dragon face-up attack level 8",
        ], opponent=True)
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == "Blue-Eyes White Dragon"
        assert gs.opp_graveyard[0].spec == "og1"

    def test_resync_grave_empty(self, gs: MUDGameState):
        gs.my_graveyard.append(CardEntry(name="stale"))
        gs.resync_grave(["No cards."])
        assert len(gs.my_graveyard) == 0

    def test_resync_grave_facedown(self, gs: MUDGameState):
        gs.resync_grave(["og1 face-down defense"], opponent=True)
        assert len(gs.opp_graveyard) == 1
        assert gs.opp_graveyard[0].name == ""
        assert gs.opp_graveyard[0].position == "face-down defense"


# ---------------------------------------------------------------------------
# Resync — removed
# ---------------------------------------------------------------------------

class TestResyncRemoved:
    def test_resync_removed_own(self, gs: MUDGameState):
        gs.resync_removed([
            "r1 Dark Magician face-up attack level 7",
        ])
        assert len(gs.my_banished) == 1
        assert gs.my_banished[0].name == "Dark Magician"
        assert gs.my_banished[0].spec == "r1"

    def test_resync_removed_opp(self, gs: MUDGameState):
        gs.resync_removed([
            "or1 Kuriboh face-up defense level 1",
        ], opponent=True)
        assert len(gs.opp_banished) == 1
        assert gs.opp_banished[0].name == "Kuriboh"

    def test_resync_removed_empty(self, gs: MUDGameState):
        gs.my_banished.append(CardEntry(name="stale"))
        gs.resync_removed(["No cards."])
        assert len(gs.my_banished) == 0


# ---------------------------------------------------------------------------
# Resync — extra
# ---------------------------------------------------------------------------

class TestResyncExtra:
    def test_resync_extra_own(self, gs: MUDGameState):
        gs.resync_extra([
            "x1 Decode Talker face-up attack link rating 3",
        ])
        assert len(gs.my_extra) == 1
        assert gs.my_extra[0].name == "Decode Talker"
        assert gs.my_extra[0].spec == "x1"

    def test_resync_extra_opp(self, gs: MUDGameState):
        gs.resync_extra([
            "ox1 Blue-Eyes White Dragon face-up attack level 8",
        ], opponent=True)
        assert len(gs.opp_extra) == 1
        assert gs.opp_extra[0].name == "Blue-Eyes White Dragon"

    def test_resync_extra_empty(self, gs: MUDGameState):
        gs.my_extra.append(CardEntry(name="stale"))
        gs.resync_extra(["No cards."])
        assert len(gs.my_extra) == 0

    def test_resync_extra_facedown_opp(self, gs: MUDGameState):
        gs.resync_extra(["ox1 face down"], opponent=True)
        assert len(gs.opp_extra) == 1
        assert gs.opp_extra[0].name == ""
        assert gs.opp_extra[0].position == "face down"


# ---------------------------------------------------------------------------
# Resync — score drift for opp GY and banished
# ---------------------------------------------------------------------------

class TestResyncScoreDrift:
    def test_opp_gy_drift(self, gs: MUDGameState, caplog):
        gs.opp_graveyard = [CardEntry(name="x")] * 3  # tracked: 3
        import logging
        with caplog.at_level(logging.WARNING):
            gs.resync_score(["Grave: You: 0 Opponent: 1"])
        assert "opp GY drift" in caplog.text

    def test_banished_drift(self, gs: MUDGameState, caplog):
        gs.my_banished = [CardEntry(name="x")] * 2  # tracked: 2
        import logging
        with caplog.at_level(logging.WARNING):
            gs.resync_score(["Removed: You: 0 Opponent: 0"])
        assert "banished drift" in caplog.text

    def test_opp_banished_drift(self, gs: MUDGameState, caplog):
        gs.opp_banished = [CardEntry(name="x")] * 2  # tracked: 2
        import logging
        with caplog.at_level(logging.WARNING):
            gs.resync_score(["Removed: You: 0 Opponent: 0"])
        assert "opp banished drift" in caplog.text


# ---------------------------------------------------------------------------
# Full event sequence (integration-style)
# ---------------------------------------------------------------------------

class TestEventSequence:
    def test_basic_duel_flow(self, gs: MUDGameState):
        """Simulate a short duel sequence and verify cumulative state."""
        # Turn 1 — mine
        gs.update(_ev(EventType.NEW_TURN, player="you"))
        gs.update(_ev(EventType.NEW_PHASE, phase="draw phase"))
        gs.update(_ev(EventType.DRAW, player="you", amount=1))
        gs.update(_ev(EventType.DRAW_CARD, player="you",
                       card_name="Mystical Space Typhoon"))
        gs.update(_ev(EventType.NEW_PHASE, phase="main1 phase"))

        # Summon Dark Magician
        gs.update(_ev(
            EventType.SUMMON, player="P1",
            card_name="Dark Magician", card_spec="m1",
            position="face-up attack"))

        assert gs.turn == 1
        assert gs.is_my_turn is True
        assert len(gs.my_hand) == 1
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Dark Magician"

        # Battle — attack, opponent takes damage
        gs.update(_ev(EventType.NEW_PHASE, phase="battle phase"))
        gs.update(_ev(
            EventType.ATTACK, player="P1",
            card_spec="m1", card_name="Dark Magician"))
        gs.update(_ev(
            EventType.DAMAGE, player="P2", is_opponent=True,
            amount=2500, new_lp=5500))

        assert gs.opp_lp == 5500
        assert gs.phase == "battle phase"

        # Turn 2 — opponent
        gs.update(_ev(EventType.NEW_TURN, player="P2", is_opponent=True))
        gs.update(_ev(EventType.NEW_PHASE, phase="draw phase"))
        gs.update(_ev(
            EventType.DRAW, player="P2", is_opponent=True, amount=1))

        assert gs.turn == 2
        assert gs.is_my_turn is False
        assert gs.opp_hand_count == 1

        # Opponent destroys Dark Magician
        gs.update(_ev(
            EventType.DESTROY, card_spec="m1", card_name="Dark Magician"))
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="m1", card_name="Dark Magician"))

        assert len(gs.my_mzone) == 0
        assert len(gs.my_graveyard) == 1
        assert gs.my_graveyard[0].name == "Dark Magician"


# ---------------------------------------------------------------------------
# S10: Clear card_spec on GY/banished entry
# ---------------------------------------------------------------------------

class TestClearSpecOnGYBanished:
    def _summon(self, gs, name, spec, opponent=False):
        gs.update(_ev(
            EventType.SUMMON, player="P2" if opponent else "P1",
            is_opponent=opponent,
            card_name=name, card_spec=spec, position="face-up attack"))
        zone = gs.opp_mzone if opponent else gs.my_mzone
        zone[-1].spec = spec

    def test_to_gy_clears_spec(self, gs: MUDGameState):
        self._summon(gs, "Dark Magician", "m1")
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="you",
            card_spec="m1", card_name="Dark Magician"))
        assert gs.my_graveyard[0].spec == ""

    def test_banished_clears_spec(self, gs: MUDGameState):
        self._summon(gs, "Kuriboh", "m2")
        gs.update(_ev(
            EventType.BANISHED, player="you",
            card_spec="m2", card_name="Kuriboh"))
        assert gs.my_banished[0].spec == ""

    def test_opp_to_gy_clears_spec(self, gs: MUDGameState):
        self._summon(gs, "Blue-Eyes White Dragon", "om1", opponent=True)
        gs.update(_ev(
            EventType.TO_GRAVEYARD, player="P2", is_opponent=True,
            card_spec="om1", card_name="Blue-Eyes White Dragon"))
        assert gs.opp_graveyard[0].spec == ""


# ---------------------------------------------------------------------------
# S11: _on_pos_change searches spell/trap zones
# ---------------------------------------------------------------------------

class TestPosChangeSzone:
    def test_pos_change_szone(self, gs: MUDGameState):
        gs.update(_ev(
            EventType.SET, player="you", card_spec="s1",
            card_name="Mirror Force", position="face-down"))
        gs.update(_ev(
            EventType.POS_CHANGE, card_spec="s1",
            card_name="Mirror Force", position="face-up"))
        assert gs.my_szone[0].position == "face-up"

    def test_pos_change_opp_szone(self, gs: MUDGameState):
        gs.opp_szone.append(CardEntry(
            name="", spec="os1", position="face-down"))
        gs.update(_ev(
            EventType.POS_CHANGE, card_spec="os1",
            card_name="Trap Card", position="face-up"))
        assert gs.opp_szone[0].position == "face-up"
        assert gs.opp_szone[0].name == "Trap Card"


# ---------------------------------------------------------------------------
# S12: Remove extra deck entry on SP_SUMMON (guarded)
# ---------------------------------------------------------------------------

class TestSpSummonExtraDeck:
    def test_sp_summon_removes_from_extra(self, gs: MUDGameState):
        gs.my_extra.append(CardEntry(name="Decode Talker", code=0, spec="x1"))
        # No copy in GY or banished → should remove from extra (name-based)
        gs.update(_ev(
            EventType.SP_SUMMON, player="Player1",
            card_name="Decode Talker", position="face-up attack"))
        assert len(gs.my_mzone) == 1
        assert gs.my_mzone[0].name == "Decode Talker"
        assert len(gs.my_extra) == 0

    def test_sp_summon_does_not_remove_if_in_gy(self, gs: MUDGameState):
        """GY revival — extra deck should NOT be depleted."""
        gs.my_extra.append(CardEntry(
            name="Dark Magician", code=46986414, spec="x1"))
        gs.my_graveyard.append(CardEntry(
            name="Dark Magician", code=46986414, spec=""))
        gs.update(_ev(
            EventType.SP_SUMMON, player="Player1",
            card_name="Dark Magician", position="face-up attack"))
        assert len(gs.my_mzone) == 1
        # Extra entry preserved (card came from GY, not extra)
        assert len(gs.my_extra) == 1

    def test_opp_sp_summon_no_extra_removal(self, gs: MUDGameState):
        """Opponent SP summon never touches our extra deck."""
        gs.opp_extra.append(CardEntry(name="Xyz Dragon", spec="ox1"))
        gs.update(_ev(
            EventType.SP_SUMMON, player="P2", is_opponent=True,
            card_name="Xyz Dragon", position="face-up attack"))
        assert len(gs.opp_mzone) == 1
        assert len(gs.opp_extra) == 1


# ---------------------------------------------------------------------------
# S13: XYZ material attach/detach
# ---------------------------------------------------------------------------

class TestXYZMaterial:
    def test_xyz_attach_removes_from_field(self, gs: MUDGameState):
        gs.my_mzone.append(CardEntry(
            name="Kuriboh", code=40640057, spec="m1"))
        gs.update(_ev(
            EventType.XYZ_ATTACH, player="you",
            card_spec="m1", card_name="Kuriboh",
            target_spec="m2", target_name="Number 39: Utopia"))
        assert len(gs.my_mzone) == 0

    def test_xyz_attach_opp(self, gs: MUDGameState):
        gs.opp_mzone.append(CardEntry(
            name="Blue-Eyes White Dragon", code=89631139, spec="om1"))
        gs.update(_ev(
            EventType.XYZ_ATTACH, player="P2", is_opponent=True,
            card_spec="om1", card_name="Blue-Eyes White Dragon",
            target_spec="om2", target_name="Galaxy-Eyes"))
        assert len(gs.opp_mzone) == 0

    def test_xyz_detach_is_noop(self, gs: MUDGameState):
        """Detach doesn't move cards — subsequent TO_GRAVEYARD does."""
        gs.update(_ev(
            EventType.XYZ_DETACH, player="you",
            card_name="Kuriboh"))
        assert len(gs.my_graveyard) == 0


# ---------------------------------------------------------------------------
# S7: resync_tab drift logging
# ---------------------------------------------------------------------------

class TestResyncTabDrift:
    def test_mzone_drift_logged(self, gs: MUDGameState, caplog):
        gs.my_mzone = [CardEntry(name="x")] * 2  # tracked: 2
        import logging
        with caplog.at_level(logging.WARNING):
            gs.resync_tab([
                "Your table:",
                "m1: Dark Magician (2500/2100) level 7 face-up attack",
            ])
        assert "mzone drift" in caplog.text
        assert "tracked 2" in caplog.text
        assert "actual 1" in caplog.text

    def test_szone_drift_logged(self, gs: MUDGameState, caplog):
        gs.my_szone = [CardEntry(name="x")] * 3  # tracked: 3
        import logging
        with caplog.at_level(logging.WARNING):
            gs.resync_tab([
                "Your table:",
                "Table is empty.",
            ])
        assert "szone drift" in caplog.text
