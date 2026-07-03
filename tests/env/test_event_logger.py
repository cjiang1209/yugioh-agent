"""Tests for the event_logger module."""

import logging

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_OVERLAY,
    LOCATION_SZONE,
    MSG_ATTACK,
    MSG_CHAIN_DISABLED,
    MSG_CHAIN_NEGATED,
    MSG_CHAINING,
    MSG_DAMAGE,
    MSG_DRAW,
    MSG_EQUIP,
    MSG_FLIPSUMMONING,
    MSG_MOVE,
    MSG_NEW_PHASE,
    MSG_NEW_TURN,
    MSG_PAY_LPCOST,
    MSG_POS_CHANGE,
    MSG_RECOVER,
    MSG_SET,
    MSG_SPSUMMONING,
    MSG_SUMMONING,
    MSG_TOSS_COIN,
    MSG_TOSS_DICE,
    PHASE_BATTLE,
    PHASE_DRAW,
    PHASE_END,
    PHASE_MAIN1,
    POS_FACEDOWN,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_env.event_logger import (
    CardInfo,
    EventDescriber,
    FieldTracker,
    enrich_messages,
    format_card,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAMES = {
    89631139: "Blue-Eyes White Dragon",
    46986414: "Dark Magician",
    5318639: "Mystical Space Typhoon",
    48800175: "The Melody of Awakening Dragon",
    38517737: "Alexandrite Dragon",
}


class _FakeCardDB:
    def get_card_name(self, code: int) -> str:
        return NAMES.get(code, f"Card#{code}")


def _fmt(messages, agent_player=0, tracker=None, sys_strings=None):
    """Shorthand: enrich then describe via an EventDescriber over the fake DB."""
    if tracker is None:
        tracker = FieldTracker()
    enriched = enrich_messages(messages, tracker)
    return EventDescriber(_FakeCardDB(), sys_strings).describe(enriched, agent_player)


# ---------------------------------------------------------------------------
# enrich_messages
# ---------------------------------------------------------------------------


class TestEnrichMessages:
    def test_attack_stamps_attacker_code(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        enriched = enrich_messages(
            [
                {
                    "msg_type": MSG_ATTACK,
                    "attacker_controller": 0,
                    "attacker_location": LOCATION_MZONE,
                    "attacker_sequence": 0,
                    "target_location": 0,
                    "target_controller": 0,
                    "target_sequence": 0,
                }
            ],
            tracker,
        )
        assert enriched[0]["attacker_code"] == 89631139

    def test_equip_stamps_both_codes(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SET,
                "code": 5318639,
                "controller": 0,
                "location": LOCATION_SZONE,
                "sequence": 0,
                "position": POS_FACEUP,
            }
        )
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        enriched = enrich_messages(
            [
                {
                    "msg_type": MSG_EQUIP,
                    "equip_controller": 0,
                    "equip_location": LOCATION_SZONE,
                    "equip_sequence": 0,
                    "target_controller": 0,
                    "target_location": LOCATION_MZONE,
                    "target_sequence": 0,
                }
            ],
            tracker,
        )
        assert enriched[0]["equip_code"] == 5318639
        assert enriched[0]["target_code"] == 89631139

    def test_passthrough_message_unchanged(self):
        tracker = FieldTracker()
        msgs = [{"msg_type": MSG_DRAW, "player": 0, "cards": [1, 2]}]
        enriched = enrich_messages(msgs, tracker)
        assert enriched[0] == {"msg_type": MSG_DRAW, "player": 0, "cards": [1, 2]}

    def test_does_not_mutate_input(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        original = {
            "msg_type": MSG_ATTACK,
            "attacker_controller": 0,
            "attacker_location": LOCATION_MZONE,
            "attacker_sequence": 0,
            "target_location": 0,
        }
        enrich_messages([original], tracker)
        assert "attacker_code" not in original  # input untouched


# ---------------------------------------------------------------------------
# format_card
# ---------------------------------------------------------------------------


class TestFormatCard:
    def test_monster_zone(self):
        result = format_card(
            89631139, "Blue-Eyes White Dragon", LOCATION_MZONE, 0, POS_FACEUP_ATTACK
        )
        assert result == "[89631139: Blue-Eyes White Dragon][MZone-0][FU-Atk]"

    def test_szone_facedown(self):
        result = format_card(
            5318639, "Mystical Space Typhoon", LOCATION_SZONE, 2, POS_FACEDOWN_DEFENSE
        )
        assert result == "[5318639: Mystical Space Typhoon][SZone-2][FD-Def]"

    def test_hand_no_position(self):
        result = format_card(
            48800175, "The Melody of Awakening Dragon", LOCATION_HAND, 0, POS_FACEUP_ATTACK
        )
        assert result == "[48800175: The Melody of Awakening Dragon][Hand]"

    def test_graveyard_no_position(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", LOCATION_GRAVE)
        assert result == "[89631139: Blue-Eyes White Dragon][GY]"

    def test_banished(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", LOCATION_BANISHED)
        assert result == "[89631139: Blue-Eyes White Dragon][Banished]"

    def test_deck(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", LOCATION_DECK)
        assert result == "[89631139: Blue-Eyes White Dragon][Deck]"

    def test_extra_deck(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", LOCATION_EXTRA)
        assert result == "[89631139: Blue-Eyes White Dragon][Extra Deck]"

    def test_overlay(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", LOCATION_OVERLAY)
        assert result == "[89631139: Blue-Eyes White Dragon][Overlay]"

    def test_unknown_location(self):
        result = format_card(89631139, "Blue-Eyes White Dragon", 0xFF)
        assert result == "[89631139: Blue-Eyes White Dragon][loc=0xff]"

    def test_pos_faceup_composite(self):
        """POS_FACEUP (0x5) should be resolved."""
        result = format_card(5318639, "MST", LOCATION_SZONE, 1, POS_FACEUP)
        assert result == "[5318639: MST][SZone-1][FU]"

    def test_pos_facedown_composite(self):
        """POS_FACEDOWN (0xA) should be resolved."""
        result = format_card(5318639, "MST", LOCATION_SZONE, 1, POS_FACEDOWN)
        assert result == "[5318639: MST][SZone-1][FD]"

    def test_szone_fallback_faceup_bitmask(self):
        """SZone with unrecognized position containing FU bits → [FU]."""
        result = format_card(5318639, "MST", LOCATION_SZONE, 0, 0x11)  # has bit 0x1
        assert "[FU]" in result

    def test_szone_fallback_facedown_bitmask(self):
        """SZone with unrecognized position containing FD bits → [FD]."""
        result = format_card(5318639, "MST", LOCATION_SZONE, 0, 0x22)  # has bit 0x2
        assert "[FD]" in result

    def test_mzone_zero_position(self):
        """MZone with position=0 → no position suffix."""
        result = format_card(89631139, "BEWD", LOCATION_MZONE, 0, 0)
        assert result == "[89631139: BEWD][MZone-0]"


# ---------------------------------------------------------------------------
# CardInfo
# ---------------------------------------------------------------------------


class TestCardInfo:
    def test_defaults(self):
        info = CardInfo()
        assert info.code == 0
        assert info.controller == 0
        assert info.location == 0
        assert info.sequence == 0
        assert info.position == 0

    def test_all_fields(self):
        info = CardInfo(89631139, 1, LOCATION_MZONE, 2, POS_FACEUP_ATTACK)
        assert info.code == 89631139
        assert info.controller == 1
        assert info.location == LOCATION_MZONE
        assert info.sequence == 2
        assert info.position == POS_FACEUP_ATTACK


# ---------------------------------------------------------------------------
# FieldTracker
# ---------------------------------------------------------------------------


class TestFieldTracker:
    def test_summon_registers_card(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        info = tracker.get(0, LOCATION_MZONE, 0)
        assert info.code == 89631139
        assert info.position == POS_FACEUP_ATTACK

    def test_sp_summon_registers(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SPSUMMONING,
                "code": 46986414,
                "controller": 1,
                "location": LOCATION_MZONE,
                "sequence": 1,
                "position": POS_FACEUP_ATTACK,
            }
        )
        info = tracker.get(1, LOCATION_MZONE, 1)
        assert info.code == 46986414

    def test_set_registers(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SET,
                "code": 5318639,
                "controller": 0,
                "location": LOCATION_SZONE,
                "sequence": 2,
                "position": POS_FACEDOWN,
            }
        )
        info = tracker.get(0, LOCATION_SZONE, 2)
        assert info.code == 5318639
        assert info.position == POS_FACEDOWN

    def test_move_updates_location(self):
        tracker = FieldTracker()
        # Summon to MZone-0
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        # Move to GY
        tracker.update(
            {
                "msg_type": MSG_MOVE,
                "code": 89631139,
                "prev_controller": 0,
                "prev_location": LOCATION_MZONE,
                "prev_sequence": 0,
                "cur_controller": 0,
                "cur_location": LOCATION_GRAVE,
                "cur_sequence": 0,
                "cur_position": 0,
            }
        )
        # Old position should be cleared
        info = tracker.get(0, LOCATION_MZONE, 0)
        assert info.code == 0  # _EMPTY_CARD
        # New position should have the card
        info = tracker.get(0, LOCATION_GRAVE, 0)
        assert info.code == 89631139

    def test_pos_change_updates_existing(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        tracker.update(
            {
                "msg_type": MSG_POS_CHANGE,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "prev_position": POS_FACEUP_ATTACK,
                "cur_position": POS_FACEUP_DEFENSE,
            }
        )
        info = tracker.get(0, LOCATION_MZONE, 0)
        assert info.position == POS_FACEUP_DEFENSE

    def test_pos_change_creates_if_missing(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_POS_CHANGE,
                "code": 46986414,
                "controller": 1,
                "location": LOCATION_MZONE,
                "sequence": 2,
                "prev_position": POS_FACEDOWN_DEFENSE,
                "cur_position": POS_FACEUP_ATTACK,
            }
        )
        info = tracker.get(1, LOCATION_MZONE, 2)
        assert info.code == 46986414
        assert info.position == POS_FACEUP_ATTACK

    def test_get_unknown_returns_empty(self):
        tracker = FieldTracker()
        info = tracker.get(0, LOCATION_MZONE, 3)
        assert info.code == 0

    def test_reset_clears_all(self):
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        tracker.reset()
        info = tracker.get(0, LOCATION_MZONE, 0)
        assert info.code == 0

    def test_zero_code_ignored(self):
        """Messages with code=0 should not update the tracker."""
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 0,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        info = tracker.get(0, LOCATION_MZONE, 0)
        assert info.code == 0


# ---------------------------------------------------------------------------
# format_events — individual message types
# ---------------------------------------------------------------------------


class TestFormatEventsNewTurn:
    def test_agent_turn(self):
        events = _fmt([{"msg_type": MSG_NEW_TURN, "player": 0}])
        assert events == ["You: Turn start"]

    def test_opponent_turn(self):
        events = _fmt([{"msg_type": MSG_NEW_TURN, "player": 1}])
        assert events == ["Opponent: Turn start"]

    def test_agent_is_player1(self):
        events = _fmt([{"msg_type": MSG_NEW_TURN, "player": 1}], agent_player=1)
        assert events == ["You: Turn start"]


class TestFormatEventsPhase:
    def test_main1(self):
        events = _fmt([{"msg_type": MSG_NEW_PHASE, "phase": PHASE_MAIN1}])
        assert events == ["Phase: Main 1"]

    def test_draw(self):
        events = _fmt([{"msg_type": MSG_NEW_PHASE, "phase": PHASE_DRAW}])
        assert events == ["Phase: Draw"]

    def test_battle(self):
        events = _fmt([{"msg_type": MSG_NEW_PHASE, "phase": PHASE_BATTLE}])
        assert events == ["Phase: Battle"]

    def test_end(self):
        events = _fmt([{"msg_type": MSG_NEW_PHASE, "phase": PHASE_END}])
        assert events == ["Phase: End"]

    def test_unknown_phase(self):
        events = _fmt([{"msg_type": MSG_NEW_PHASE, "phase": 0xFF}])
        assert events == ["Phase: 0xff"]


class TestFormatEventsDraw:
    def test_draw_one(self):
        events = _fmt([{"msg_type": MSG_DRAW, "player": 0, "cards": [89631139]}])
        assert events == ["You: Draw 1 card"]

    def test_draw_multiple(self):
        events = _fmt([{"msg_type": MSG_DRAW, "player": 1, "cards": [1, 2, 3]}])
        assert events == ["Opponent: Draw 3 cards"]

    def test_draw_zero(self):
        events = _fmt([{"msg_type": MSG_DRAW, "player": 0, "cards": []}])
        assert events == ["You: Draw 0 cards"]


class TestFormatEventsSummon:
    def test_normal_summon(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_SUMMONING,
                    "code": 89631139,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "position": POS_FACEUP_ATTACK,
                }
            ]
        )
        assert len(events) == 1
        assert events[0] == "You: Normal Summon [89631139: Blue-Eyes White Dragon][MZone-0][FU-Atk]"

    def test_special_summon_opponent(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_SPSUMMONING,
                    "code": 46986414,
                    "controller": 1,
                    "location": LOCATION_MZONE,
                    "sequence": 1,
                    "position": POS_FACEUP_ATTACK,
                }
            ]
        )
        assert "Opponent: Special Summon" in events[0]
        assert "[46986414: Dark Magician]" in events[0]

    def test_flip_summon(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_FLIPSUMMONING,
                    "code": 38517737,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 2,
                    "position": POS_FACEUP_ATTACK,
                }
            ]
        )
        assert "You: Flip Summon" in events[0]
        assert "[38517737: Alexandrite Dragon]" in events[0]


class TestFormatEventsChaining:
    def test_activate(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_CHAINING,
                    "code": 5318639,
                    "controller": 1,
                    "location": LOCATION_SZONE,
                    "sequence": 0,
                    "position": POS_FACEUP,
                }
            ]
        )
        assert events[0] == "Opponent: Activate [5318639: Mystical Space Typhoon][SZone-0][FU]"

    def test_activate_includes_chain_link(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_CHAINING,
                    "code": 5318639,
                    "controller": 1,
                    "location": LOCATION_SZONE,
                    "sequence": 0,
                    "position": POS_FACEUP,
                    "chain_link": 2,
                }
            ]
        )
        assert events[0] == (
            "Opponent: [Chain 2] Activate [5318639: Mystical Space Typhoon][SZone-0][FU]"
        )

    def test_chain_negated(self):
        events = _fmt([{"msg_type": MSG_CHAIN_NEGATED}])
        assert events == ["Chain is negated"]

    def test_chain_negated_includes_chain_link(self):
        events = _fmt([{"msg_type": MSG_CHAIN_NEGATED, "chain_link": 2}])
        assert events == ["[Chain 2] negated"]

    def test_chain_disabled(self):
        events = _fmt([{"msg_type": MSG_CHAIN_DISABLED}])
        assert events == ["Chain is disabled"]

    def test_chain_disabled_includes_chain_link(self):
        events = _fmt([{"msg_type": MSG_CHAIN_DISABLED, "chain_link": 3}])
        assert events == ["[Chain 3] disabled"]


class TestFormatEventsAttack:
    def test_direct_attack(self):
        tracker = FieldTracker()
        # Register attacker first
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        events = _fmt(
            [
                {
                    "msg_type": MSG_ATTACK,
                    "attacker_controller": 0,
                    "attacker_location": LOCATION_MZONE,
                    "attacker_sequence": 0,
                    "target_location": 0,
                    "target_controller": 0,
                    "target_sequence": 0,
                }
            ],
            0,
            tracker=tracker,
        )
        assert len(events) == 1
        assert "You:" in events[0]
        assert "attacks directly" in events[0]
        assert "[89631139: Blue-Eyes White Dragon]" in events[0]

    def test_attack_target(self):
        tracker = FieldTracker()
        # Register both monsters
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 38517737,
                "controller": 1,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        events = _fmt(
            [
                {
                    "msg_type": MSG_ATTACK,
                    "attacker_controller": 0,
                    "attacker_location": LOCATION_MZONE,
                    "attacker_sequence": 0,
                    "target_controller": 1,
                    "target_location": LOCATION_MZONE,
                    "target_sequence": 0,
                }
            ],
            0,
            tracker=tracker,
        )
        assert "attacks" in events[0]
        assert "[89631139: Blue-Eyes White Dragon]" in events[0]
        assert "[38517737: Alexandrite Dragon]" in events[0]

    def test_attack_unknown_card(self, caplog):
        """Attack from a location the tracker never recorded (reconstruction
        drift): the code is unknown ([0: ?]) but the engine-provided location is
        preserved, so the line still shows the attacker's zone. The drift is
        logged as a warning, not let go silently."""
        tracker = FieldTracker()
        with caplog.at_level(logging.WARNING, logger="yugioh_env.event_logger"):
            events = _fmt(
                [
                    {
                        "msg_type": MSG_ATTACK,
                        "attacker_controller": 0,
                        "attacker_location": LOCATION_MZONE,
                        "attacker_sequence": 0,
                        "target_location": 0,
                    }
                ],
                0,
                tracker=tracker,
            )
        assert events[0] == "You: [0: ?][MZone-0] attacks directly"
        assert any("drift" in r.message.lower() for r in caplog.records)

    def test_attack_known_card_no_drift_warning(self, caplog):
        """A tracker hit must NOT emit a drift warning."""
        tracker = FieldTracker()
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        with caplog.at_level(logging.WARNING, logger="yugioh_env.event_logger"):
            _fmt(
                [
                    {
                        "msg_type": MSG_ATTACK,
                        "attacker_controller": 0,
                        "attacker_location": LOCATION_MZONE,
                        "attacker_sequence": 0,
                        "target_location": 0,
                    }
                ],
                0,
                tracker=tracker,
            )
        assert not any("drift" in r.message.lower() for r in caplog.records)


class TestFormatEventsLPChanges:
    def test_damage(self):
        events = _fmt([{"msg_type": MSG_DAMAGE, "player": 1, "amount": 2500}])
        assert events == ["Opponent: Take 2500 damage"]

    def test_recover(self):
        events = _fmt([{"msg_type": MSG_RECOVER, "player": 0, "amount": 500}])
        assert events == ["You: Recover 500 LP"]

    def test_pay_lp(self):
        events = _fmt([{"msg_type": MSG_PAY_LPCOST, "player": 0, "amount": 800}])
        assert events == ["You: Pay 800 LP"]


class TestFormatEventsMove:
    def test_to_graveyard(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 0,
                    "prev_location": LOCATION_MZONE,
                    "prev_sequence": 0,
                    "prev_position": POS_FACEUP_ATTACK,
                    "cur_controller": 0,
                    "cur_location": LOCATION_GRAVE,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is sent to Graveyard" in events[0]
        assert "You:" in events[0]

    def test_to_banished(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 46986414,
                    "prev_controller": 1,
                    "prev_location": LOCATION_GRAVE,
                    "prev_sequence": 0,
                    "prev_position": 0,
                    "cur_controller": 1,
                    "cur_location": LOCATION_BANISHED,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is banished" in events[0]
        assert "Opponent:" in events[0]

    def test_to_deck(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 0,
                    "prev_location": LOCATION_HAND,
                    "prev_sequence": 0,
                    "prev_position": 0,
                    "cur_controller": 0,
                    "cur_location": LOCATION_DECK,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is returned to Deck" in events[0]
        assert "You:" in events[0]

    def test_deck_to_hand(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 48800175,
                    "prev_controller": 0,
                    "prev_location": LOCATION_DECK,
                    "prev_sequence": 0,
                    "prev_position": 0,
                    "cur_controller": 0,
                    "cur_location": LOCATION_HAND,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is added to Hand" in events[0]
        assert "You:" in events[0]

    def test_field_to_hand(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 0,
                    "prev_location": LOCATION_MZONE,
                    "prev_sequence": 0,
                    "prev_position": POS_FACEUP_ATTACK,
                    "cur_controller": 0,
                    "cur_location": LOCATION_HAND,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is returned to Hand" in events[0]
        assert "You:" in events[0]

    def test_gy_to_banished(self):
        """GY → Banished should say 'is banished'."""
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 0,
                    "prev_location": LOCATION_GRAVE,
                    "prev_sequence": 0,
                    "prev_position": 0,
                    "cur_controller": 0,
                    "cur_location": LOCATION_BANISHED,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is banished" in events[0]

    def test_hand_to_gy(self):
        """Hand → GY should say 'is sent to Graveyard'."""
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 5318639,
                    "prev_controller": 1,
                    "prev_location": LOCATION_HAND,
                    "prev_sequence": 3,
                    "prev_position": 0,
                    "cur_controller": 1,
                    "cur_location": LOCATION_GRAVE,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert "is sent to Graveyard" in events[0]
        assert "Opponent:" in events[0]

    def test_move_to_deck_uses_cur_controller(self):
        """Return-to-deck uses current controller (may differ from prev)."""
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 1,
                    "prev_location": LOCATION_MZONE,
                    "prev_sequence": 0,
                    "prev_position": POS_FACEUP_ATTACK,
                    "cur_controller": 0,
                    "cur_location": LOCATION_DECK,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        # cur_controller=0 → "You:"
        assert "You:" in events[0]
        assert "is returned to Deck" in events[0]

    def test_move_unhandled_destination(self):
        """Move to a location not explicitly handled → no event."""
        events = _fmt(
            [
                {
                    "msg_type": MSG_MOVE,
                    "code": 89631139,
                    "prev_controller": 0,
                    "prev_location": LOCATION_HAND,
                    "prev_sequence": 0,
                    "prev_position": 0,
                    "cur_controller": 0,
                    "cur_location": LOCATION_EXTRA,
                    "cur_sequence": 0,
                    "cur_position": 0,
                }
            ]
        )
        assert events == []


class TestFormatEventsPosChange:
    def test_atk_to_def(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_POS_CHANGE,
                    "code": 89631139,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "prev_position": POS_FACEUP_ATTACK,
                    "cur_position": POS_FACEUP_DEFENSE,
                }
            ]
        )
        assert "changes position to FU-Def" in events[0]
        # Card info should show *previous* position
        assert "[FU-Atk]" in events[0]


class TestFormatEventsSet:
    def test_set(self):
        events = _fmt(
            [
                {
                    "msg_type": MSG_SET,
                    "code": 5318639,
                    "controller": 0,
                    "location": LOCATION_SZONE,
                    "sequence": 3,
                    "position": POS_FACEDOWN,
                }
            ]
        )
        assert events[0] == "You: Set [5318639: Mystical Space Typhoon][SZone-3][FD]"


class TestFormatEventsEquip:
    def test_equip(self):
        tracker = FieldTracker()
        # Register equip card and target
        tracker.update(
            {
                "msg_type": MSG_SET,
                "code": 5318639,
                "controller": 0,
                "location": LOCATION_SZONE,
                "sequence": 0,
                "position": POS_FACEUP,
            }
        )
        tracker.update(
            {
                "msg_type": MSG_SUMMONING,
                "code": 89631139,
                "controller": 0,
                "location": LOCATION_MZONE,
                "sequence": 0,
                "position": POS_FACEUP_ATTACK,
            }
        )
        events = _fmt(
            [
                {
                    "msg_type": MSG_EQUIP,
                    "equip_controller": 0,
                    "equip_location": LOCATION_SZONE,
                    "equip_sequence": 0,
                    "target_controller": 0,
                    "target_location": LOCATION_MZONE,
                    "target_sequence": 0,
                }
            ],
            0,
            tracker=tracker,
        )
        assert "is equipped to" in events[0]
        assert "[5318639: Mystical Space Typhoon]" in events[0]
        assert "[89631139: Blue-Eyes White Dragon]" in events[0]


class TestFormatEventsToss:
    def test_coin_heads(self):
        events = _fmt([{"msg_type": MSG_TOSS_COIN, "results": [1]}])
        assert events == ["Coin toss: Heads"]

    def test_coin_tails(self):
        events = _fmt([{"msg_type": MSG_TOSS_COIN, "results": [0]}])
        assert events == ["Coin toss: Tails"]

    def test_coin_multiple(self):
        events = _fmt([{"msg_type": MSG_TOSS_COIN, "results": [1, 0, 1]}])
        assert events == ["Coin toss: Heads, Tails, Heads"]

    def test_dice_single(self):
        events = _fmt([{"msg_type": MSG_TOSS_DICE, "results": [4]}])
        assert events == ["Dice roll: 4"]

    def test_dice_multiple(self):
        events = _fmt([{"msg_type": MSG_TOSS_DICE, "results": [3, 6]}])
        assert events == ["Dice roll: 3, 6"]


# ---------------------------------------------------------------------------
# format_events — unhandled & multi-message
# ---------------------------------------------------------------------------


class TestFormatEventsEdgeCases:
    def test_unknown_msg_type_ignored(self):
        """Messages with unrecognized types produce no events."""
        events = _fmt([{"msg_type": 9999}])
        assert events == []

    def test_empty_messages(self):
        events = _fmt([])
        assert events == []

    def test_multiple_messages(self):
        """Multiple messages produce multiple events in order."""
        messages = [
            {"msg_type": MSG_NEW_TURN, "player": 0},
            {"msg_type": MSG_NEW_PHASE, "phase": PHASE_DRAW},
            {"msg_type": MSG_DRAW, "player": 0, "cards": [89631139]},
            {"msg_type": MSG_NEW_PHASE, "phase": PHASE_MAIN1},
        ]
        events = _fmt(messages)
        assert len(events) == 4
        assert events[0] == "You: Turn start"
        assert events[1] == "Phase: Draw"
        assert events[2] == "You: Draw 1 card"
        assert events[3] == "Phase: Main 1"


# ---------------------------------------------------------------------------
# FieldTracker persistence across format_events calls
# ---------------------------------------------------------------------------


class TestFieldTrackerPersistence:
    def test_tracker_persists_across_calls(self):
        """Tracker retains card info across multiple format_events calls."""
        tracker = FieldTracker()
        # First call: summon a monster
        _fmt(
            [
                {
                    "msg_type": MSG_SUMMONING,
                    "code": 89631139,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "position": POS_FACEUP_ATTACK,
                }
            ],
            tracker=tracker,
        )

        # Second call: attack with that monster (no summon in this batch)
        events = _fmt(
            [
                {
                    "msg_type": MSG_ATTACK,
                    "attacker_controller": 0,
                    "attacker_location": LOCATION_MZONE,
                    "attacker_sequence": 0,
                    "target_location": 0,
                }
            ],
            0,
            tracker=tracker,
        )
        assert "[89631139: Blue-Eyes White Dragon]" in events[0]

    def test_tracker_reset_clears_persistence(self):
        """After reset, previously tracked cards are gone."""
        tracker = FieldTracker()
        _fmt(
            [
                {
                    "msg_type": MSG_SUMMONING,
                    "code": 89631139,
                    "controller": 0,
                    "location": LOCATION_MZONE,
                    "sequence": 0,
                    "position": POS_FACEUP_ATTACK,
                }
            ],
            tracker=tracker,
        )

        tracker.reset()

        events = _fmt(
            [
                {
                    "msg_type": MSG_ATTACK,
                    "attacker_controller": 0,
                    "attacker_location": LOCATION_MZONE,
                    "attacker_sequence": 0,
                    "target_location": 0,
                }
            ],
            0,
            tracker=tracker,
        )
        # Card code should be 0 (unknown) after reset
        assert "[0: ?]" in events[0]


# ---------------------------------------------------------------------------
# Chain effect text — resolver integration
# ---------------------------------------------------------------------------


def test_chaining_with_resolver_appends_effect_text():
    # desc=12345 -> passcode=0, n=12345 (a system string)
    events = _fmt(
        [
            {
                "msg_type": MSG_CHAINING,
                "code": 5318639,
                "controller": 1,
                "location": LOCATION_SZONE,
                "sequence": 0,
                "position": POS_FACEUP,
                "desc": 12345,
                "chain_link": 1,
            }
        ],
        sys_strings={12345: "Negate the summon"},
    )
    assert events[0] == (
        "Opponent: [Chain 1] Activate [5318639: Mystical Space Typhoon][SZone-0][FU]: "
        "Negate the summon"
    )


def test_chaining_without_resolver_is_name_only():
    events = _fmt(
        [
            {
                "msg_type": MSG_CHAINING,
                "code": 5318639,
                "controller": 1,
                "location": LOCATION_SZONE,
                "sequence": 0,
                "position": POS_FACEUP,
                "desc": 12345,
            }
        ]
    )
    assert events[0] == "Opponent: Activate [5318639: Mystical Space Typhoon][SZone-0][FU]"
