"""Roundtrip tests: encode (observation.py / action_space.py) → decode (features.py).

Every meaningful field written by the encoder must survive decoding with
the correct value.  These tests catch silent field drops (like the race
bug) and byte-offset mismatches.
"""

import numpy as np
import pytest
import torch

from tests.env.conftest import action_features
from yugioh_core.constants import (
    ATTRIBUTE_DARK,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_SELECT_CARD,
    MSG_SELECT_IDLECMD,
    PHASE_MAIN1,
    POS_FACEUP_ATTACK,
    RACE_DRAGON,
    RACE_MACHINE,
)
from yugioh_core.encoding import (
    CARD_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    encode_card,
)
from yugioh_env.action_space import ActionMapper
from yugioh_env.game_state import GameState
from yugioh_env.models import YuGiOhObservation
from yugioh_env.observation import build_observation
from yugioh_rl.features import (
    _ATTR_BITS,
    _LINK_BITS,
    _LOC_BITS,
    _PHASE_BITS,
    _RACE_BITS,
    _TYPE_BITS,
    ACTION_FEAT_DIM,
    CARD_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    decode_actions,
    decode_cards,
    decode_global,
)
from yugioh_rl.obs_encoder import encode_observation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card_tensor(card_array: np.ndarray) -> torch.Tensor:
    """Wrap a single encoded card in a (1, MAX_CARDS, CARD_FEATURES) batch tensor."""
    cards = np.zeros((1, MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
    cards[0, 0] = card_array
    return torch.from_numpy(cards)


def _global_tensor(gs: GameState, msg=None, agent_player=0) -> torch.Tensor:
    """Build an observation and return its packed global row as a
    (1, GLOBAL_FEATURES) tensor."""
    obs_data = build_observation(gs, msg, agent_player=agent_player)
    obs = YuGiOhObservation(global_state=obs_data["global_state"])
    return torch.from_numpy(encode_observation(obs)["global_state"]).unsqueeze(0)


def _bit_index(bits: list[int], value: int) -> list[int]:
    """Return which bit positions are set for *value* in *bits*."""
    return [i for i, b in enumerate(bits) if value & b]


# ---------------------------------------------------------------------------
# Card feature roundtrips
# ---------------------------------------------------------------------------


class TestCardRoundtrip:
    """encode_card → decode_cards roundtrips."""

    def test_output_shape(self):
        card = encode_card(
            code=100,
            location=LOCATION_HAND,
            sequence=0,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
        )
        assert card.shape == (CARD_FEATURES,)
        assert card.dtype == np.uint8

        ids, feats = decode_cards(_card_tensor(card))
        assert ids.shape == (1, MAX_CARDS)
        assert feats.shape == (1, MAX_CARDS, CARD_FEAT_DIM)

    def test_card_id(self):
        card = encode_card(
            code=46986414,
            location=LOCATION_HAND,
            sequence=0,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
        )
        ids, _ = decode_cards(_card_tensor(card))
        assert ids[0, 0].item() == 46986414

    def test_card_id_large(self):
        """Card IDs > 65535 survive the uint32 roundtrip."""
        code = 100000123
        card = encode_card(
            code=code,
            location=LOCATION_HAND,
            sequence=0,
            position=POS_FACEUP_ATTACK,
            controller=0,
            is_public=True,
        )
        ids, _ = decode_cards(_card_tensor(card))
        assert ids[0, 0].item() == code

    def test_location(self):
        for loc, expected_bits in [
            (LOCATION_HAND, [0]),  # 0x02 → _LOC_BITS[0]
            (LOCATION_MZONE, [1]),  # 0x04 → _LOC_BITS[1]
        ]:
            card = encode_card(
                code=0,
                location=loc,
                sequence=0,
                position=0,
                controller=0,
                is_public=False,
            )
            _, feats = decode_cards(_card_tensor(card))
            loc_feats = feats[0, 0, :7]
            for i in range(7):
                if i in expected_bits:
                    assert loc_feats[i].item() == 1.0, (
                        f"loc bit {i} should be set for location 0x{loc:02x}"
                    )
                else:
                    assert loc_feats[i].item() == 0.0, (
                        f"loc bit {i} should be clear for location 0x{loc:02x}"
                    )

    def test_sequence(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=5,
            position=0,
            controller=0,
            is_public=False,
        )
        _, feats = decode_cards(_card_tensor(card))
        assert feats[0, 0, 7].item() == pytest.approx(5 / 15.0)

    def test_position(self):
        for pos_val, expected_idx in [
            (0x01, 0),  # FU-Atk
            (0x04, 2),  # FU-Def
            (0x08, 3),  # FD-Def
        ]:
            card = encode_card(
                code=0,
                location=LOCATION_MZONE,
                sequence=0,
                position=pos_val,
                controller=0,
                is_public=False,
            )
            _, feats = decode_cards(_card_tensor(card))
            pos_feats = feats[0, 0, 8:12]
            for i in range(4):
                expected = 1.0 if i == expected_idx else 0.0
                assert pos_feats[i].item() == expected

    def test_controller(self):
        for ctrl in [0, 1]:
            card = encode_card(
                code=0,
                location=LOCATION_MZONE,
                sequence=0,
                position=0,
                controller=ctrl,
                is_public=False,
            )
            _, feats = decode_cards(_card_tensor(card))
            assert feats[0, 0, 12].item() == float(ctrl)

    def test_is_public(self):
        for pub in [True, False]:
            card = encode_card(
                code=0,
                location=LOCATION_MZONE,
                sequence=0,
                position=0,
                controller=0,
                is_public=pub,
            )
            _, feats = decode_cards(_card_tensor(card))
            assert feats[0, 0, 13].item() == (1.0 if pub else 0.0)

    def test_card_type(self):
        # Monster + Effect = 0x21
        ctype = 0x21
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            card_type=ctype,
        )
        _, feats = decode_cards(_card_tensor(card))
        n_type_bits = len(_TYPE_BITS)
        type_feats = feats[0, 0, 14 : 14 + n_type_bits]
        expected_set = _bit_index(_TYPE_BITS, ctype)
        for i in range(n_type_bits):
            expected = 1.0 if i in expected_set else 0.0
            assert type_feats[i].item() == expected, f"type bit {i}"

    def test_card_type_high_bits(self):
        # Link monster = 0x4000000 | 0x1 | 0x20 = 0x4000021
        ctype = 0x4000021
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            card_type=ctype,
        )
        _, feats = decode_cards(_card_tensor(card))
        n_type_bits = len(_TYPE_BITS)
        type_feats = feats[0, 0, 14 : 14 + n_type_bits]
        expected_set = _bit_index(_TYPE_BITS, ctype)
        for i in range(n_type_bits):
            expected = 1.0 if i in expected_set else 0.0
            assert type_feats[i].item() == expected, f"type bit {i}"

    def test_card_type_all_new_flags(self):
        """All 26 type flags decode correctly."""
        for i, bit in enumerate(_TYPE_BITS):
            card = encode_card(
                code=0,
                location=LOCATION_MZONE,
                sequence=0,
                position=0,
                controller=0,
                is_public=False,
                card_type=bit,
            )
            _, feats = decode_cards(_card_tensor(card))
            n_type_bits = len(_TYPE_BITS)
            type_feats = feats[0, 0, 14 : 14 + n_type_bits]
            assert type_feats[i].item() == 1.0, f"type bit {i} (0x{bit:x}) not set"
            # All other bits should be 0
            for j in range(n_type_bits):
                if j != i:
                    assert type_feats[j].item() == 0.0, (
                        f"type bit {j} should be 0 when only bit {i} set"
                    )

    def test_level(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            level=8,
        )
        _, feats = decode_cards(_card_tensor(card))
        # level is at offset 14 + 26 = 40
        assert feats[0, 0, 40].item() == pytest.approx(8 / 12.0)

    def test_attribute(self):
        # DARK = 0x20
        attr = 0x20
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            attribute=attr,
        )
        _, feats = decode_cards(_card_tensor(card))
        # attribute at offset 41-47
        attr_feats = feats[0, 0, 41:48]
        expected_set = _bit_index(_ATTR_BITS, attr)
        for i in range(7):
            expected = 1.0 if i in expected_set else 0.0
            assert attr_feats[i].item() == expected, f"attr bit {i}"

    def test_race(self):
        # Dragon = 0x2000
        race = RACE_DRAGON
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            race=race,
        )
        _, feats = decode_cards(_card_tensor(card))
        n_race_bits = len(_RACE_BITS)
        # race at offset 48-79
        race_feats = feats[0, 0, 48 : 48 + n_race_bits]
        expected_set = _bit_index(_RACE_BITS, race)
        for i in range(n_race_bits):
            expected = 1.0 if i in expected_set else 0.0
            assert race_feats[i].item() == expected, f"race bit {i}"

    def test_race_multi_bits(self):
        # Warrior(0x1) | Machine(0x20) — unusual, but tests multi-bit decode
        race = 0x0021
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            race=race,
        )
        _, feats = decode_cards(_card_tensor(card))
        n_race_bits = len(_RACE_BITS)
        race_feats = feats[0, 0, 48 : 48 + n_race_bits]
        expected_set = _bit_index(_RACE_BITS, race)
        assert len(expected_set) == 2
        for i in range(n_race_bits):
            expected = 1.0 if i in expected_set else 0.0
            assert race_feats[i].item() == expected, f"race bit {i}"

    def test_race_high_bits(self):
        """Race bits 16-31 (e.g., Dinosaur=0x10000, Cyberse=0x1000000) survive roundtrip."""
        for i, bit in enumerate(_RACE_BITS):
            if bit <= 0x8000:
                continue  # skip low bits, tested elsewhere
            card = encode_card(
                code=0,
                location=LOCATION_MZONE,
                sequence=0,
                position=0,
                controller=0,
                is_public=False,
                race=bit,
            )
            _, feats = decode_cards(_card_tensor(card))
            n_race_bits = len(_RACE_BITS)
            race_feats = feats[0, 0, 48 : 48 + n_race_bits]
            assert race_feats[i].item() == 1.0, f"race bit {i} (0x{bit:x}) not set"
            for j in range(n_race_bits):
                if j != i:
                    assert race_feats[j].item() == 0.0, (
                        f"race bit {j} should be 0 when only bit {i} set"
                    )

    def test_atk(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            attack=3000,
        )
        _, feats = decode_cards(_card_tensor(card))
        # ATK at offset 80
        assert feats[0, 0, 80].item() == pytest.approx(3000 / 5000.0)

    def test_def(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            defense=2500,
        )
        _, feats = decode_cards(_card_tensor(card))
        # DEF at offset 81
        assert feats[0, 0, 81].item() == pytest.approx(2500 / 5000.0)

    def test_lscale(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            lscale=4,
        )
        _, feats = decode_cards(_card_tensor(card))
        # lscale at offset 82
        assert feats[0, 0, 82].item() == pytest.approx(4 / 12.0)

    def test_rscale(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            rscale=8,
        )
        _, feats = decode_cards(_card_tensor(card))
        # rscale at offset 83
        assert feats[0, 0, 83].item() == pytest.approx(8 / 12.0)

    def test_link_marker(self):
        # Bottom-left(0x40) + bottom(0x80) = 0xC0
        lmark = 0xC0
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            link_marker=lmark,
        )
        _, feats = decode_cards(_card_tensor(card))
        # link_marker at offset 84-91
        link_feats = feats[0, 0, 84:92]
        expected_set = _bit_index(_LINK_BITS, lmark)
        for i in range(8):
            expected = 1.0 if i in expected_set else 0.0
            assert link_feats[i].item() == expected, f"link bit {i}"

    def test_counter_count(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            counter_count=3,
        )
        _, feats = decode_cards(_card_tensor(card))
        # counter at offset 92
        assert feats[0, 0, 92].item() == pytest.approx(3 / 10.0)

    def test_negated(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            negated=True,
        )
        _, feats = decode_cards(_card_tensor(card))
        # negated at offset 93
        assert feats[0, 0, 93].item() == 1.0

    def test_is_overlay(self):
        card = encode_card(
            code=0,
            location=LOCATION_MZONE,
            sequence=0,
            position=0,
            controller=0,
            is_public=False,
            is_overlay=True,
        )
        _, feats = decode_cards(_card_tensor(card))
        # is_overlay at offset 94
        assert feats[0, 0, 94].item() == 1.0

    def test_all_fields_combined(self):
        """Encode a fully-populated card and verify every field decodes."""
        card = encode_card(
            code=46986414,
            location=LOCATION_MZONE,  # MZONE
            sequence=3,
            position=POS_FACEUP_ATTACK,  # FU-Atk
            controller=1,
            is_public=True,
            card_type=0x4000021,  # link + monster + effect
            level=4,
            attribute=ATTRIBUTE_DARK,  # DARK
            race=RACE_MACHINE,  # machine
            attack=2500,
            defense=2000,
            lscale=3,
            rscale=7,
            link_marker=0x41,  # top-left(0x01) + bottom-left(0x40)
            counter_count=2,
            negated=True,
            is_overlay=True,
        )
        ids, feats = decode_cards(_card_tensor(card))
        f = feats[0, 0]

        # card_id (full uint32 now)
        assert ids[0, 0].item() == 46986414
        # location: MZONE = 0x04 → bit 1
        assert f[1].item() == 1.0
        # sequence
        assert f[7].item() == pytest.approx(3 / 15.0)
        # position: FU-Atk → bit 0
        assert f[8].item() == 1.0
        # controller
        assert f[12].item() == 1.0
        # is_public
        assert f[13].item() == 1.0
        # card_type: monster(bit0=0x1) + effect(bit4=0x20) + link(bit25=0x4000000)
        assert f[14].item() == 1.0  # monster (0x1)
        assert f[18].item() == 1.0  # effect (0x20)
        assert f[39].item() == 1.0  # link (0x4000000)
        # level
        assert f[40].item() == pytest.approx(4 / 12.0)
        # attribute: DARK → bit 5
        assert f[46].item() == 1.0
        # race: machine = 0x20 → bit 5
        assert f[53].item() == 1.0
        # ATK
        assert f[80].item() == pytest.approx(2500 / 5000.0)
        # DEF
        assert f[81].item() == pytest.approx(2000 / 5000.0)
        # lscale
        assert f[82].item() == pytest.approx(3 / 12.0)
        # rscale
        assert f[83].item() == pytest.approx(7 / 12.0)
        # link_marker: top-left(0x01→bit0) + bottom-left(0x40→bit5)
        assert f[84].item() == 1.0
        assert f[89].item() == 1.0
        # counter_count
        assert f[92].item() == pytest.approx(2 / 10.0)
        # negated
        assert f[93].item() == 1.0
        # is_overlay
        assert f[94].item() == 1.0

    def test_hidden_card_zeros(self):
        """A hidden card should decode to all-zero features (except location/controller)."""
        card = encode_card(
            code=0,
            location=LOCATION_SZONE,
            sequence=2,
            position=0,
            controller=1,
            is_public=False,
        )
        _, feats = decode_cards(_card_tensor(card))
        f = feats[0, 0]

        # location (szone = 0x08 → bit 2) and controller should be set
        assert f[2].item() == 1.0  # szone bit
        assert f[12].item() == 1.0  # controller=1
        # Everything stat-related should be zero
        for i in [13, 40, 80, 81, 82, 83, 92, 93, 94]:
            assert f[i].item() == 0.0, f"feat[{i}] should be 0 for hidden card"


# ---------------------------------------------------------------------------
# Global state roundtrips
# ---------------------------------------------------------------------------


class TestGlobalRoundtrip:
    """build_observation(global_state) → decode_global roundtrips."""

    def test_output_shape(self):
        gs = GameState()
        feats = decode_global(_global_tensor(gs))
        assert feats.shape == (1, GLOBAL_FEAT_DIM)

    def test_lp(self):
        gs = GameState()
        gs.lp = [6000, 3500]
        feats = decode_global(_global_tensor(gs))
        assert feats[0, 0].item() == pytest.approx(6000 / 8000.0)
        assert feats[0, 1].item() == pytest.approx(3500 / 8000.0)

    def test_turn_count(self):
        gs = GameState()
        gs.turn_count = 10
        feats = decode_global(_global_tensor(gs))
        assert feats[0, 2].item() == pytest.approx(10 / 50.0)

    def test_phase(self):
        # Main phase 1 = 0x04
        gs = GameState()
        gs.phase = PHASE_MAIN1
        feats = decode_global(_global_tensor(gs))
        phase_feats = feats[0, 3:13]
        expected_set = _bit_index(_PHASE_BITS, PHASE_MAIN1)
        for i in range(10):
            expected = 1.0 if i in expected_set else 0.0
            assert phase_feats[i].item() == expected, f"phase bit {i}"

    def test_is_my_turn(self):
        gs = GameState()
        gs.current_player = 0
        feats = decode_global(_global_tensor(gs, agent_player=0))
        assert feats[0, 13].item() == 1.0

        feats = decode_global(_global_tensor(gs, agent_player=1))
        assert feats[0, 13].item() == 0.0

    def test_chain_count(self):
        gs = GameState()
        gs.chain_count = 3
        feats = decode_global(_global_tensor(gs))
        assert feats[0, 14].item() == pytest.approx(3 / 5.0)

    def test_zone_counts(self):
        gs = GameState()
        gs.deck_count = [35, 30]
        gs.hand_count = [5, 6]
        gs.grave_count = [3, 4]
        gs.banished_count = [1, 2]
        gs.extra_count = [15, 10]
        feats = decode_global(_global_tensor(gs, agent_player=0))
        # Bytes 10-19: [my_deck, my_hand, my_grave, my_banished, my_extra,
        #               opp_deck, opp_hand, opp_grave, opp_banished, opp_extra]
        expected = [35, 5, 3, 1, 15, 30, 6, 4, 2, 10]
        for i, val in enumerate(expected):
            assert feats[0, 15 + i].item() == pytest.approx(val / 40.0), f"zone count {i}"

    def test_lp_swap_for_opponent_perspective(self):
        """When agent_player=1, my_lp should be player 1's LP."""
        gs = GameState()
        gs.lp = [8000, 4000]
        feats = decode_global(_global_tensor(gs, agent_player=1))
        assert feats[0, 0].item() == pytest.approx(4000 / 8000.0)  # my_lp
        assert feats[0, 1].item() == pytest.approx(8000 / 8000.0)  # opp_lp


# ---------------------------------------------------------------------------
# Action feature roundtrips
# ---------------------------------------------------------------------------


class TestActionRoundtrip:
    """encode_observation's action rows → decode_actions roundtrips."""

    def _action_tensor(self, mapper: ActionMapper) -> torch.Tensor:
        return torch.from_numpy(action_features(mapper)).unsqueeze(0)

    def test_output_shape(self):
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_CARD,
                "player": 0,
                "cancelable": 0,
                "min": 1,
                "max": 1,
                "cards": [
                    {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0}
                ],
            }
        )
        codes, desc_pcs, desc_ns, feats = decode_actions(self._action_tensor(mapper))
        assert codes.shape == (1, MAX_ACTIONS)
        assert desc_pcs.shape == (1, MAX_ACTIONS)
        assert desc_ns.shape == (1, MAX_ACTIONS)
        assert feats.shape == (1, MAX_ACTIONS, ACTION_FEAT_DIM)

    def test_action_code(self):
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_CARD,
                "player": 0,
                "cancelable": 0,
                "min": 1,
                "max": 1,
                "cards": [
                    {
                        "code": 89631139,
                        "controller": 0,
                        "location": 2,
                        "sequence": 0,
                        "subsequence": 0,
                    }
                ],
            }
        )
        codes, *_ = decode_actions(self._action_tensor(mapper))
        assert codes[0, 0].item() == 89631139

    def test_msg_type(self):
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_CARD,
                "player": 0,
                "cancelable": 0,
                "min": 1,
                "max": 1,
                "cards": [
                    {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0}
                ],
            }
        )
        *_, feats = decode_actions(self._action_tensor(mapper))
        assert feats[0, 0, 0].item() == pytest.approx(MSG_SELECT_CARD / 255.0)

    def test_category(self):
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_IDLECMD,
                "player": 0,
                "summonable": [{"code": 100, "location": LOCATION_HAND, "sequence": 0}],
                "sp_summonable": [{"code": 200, "location": LOCATION_HAND, "sequence": 1}],
                "repositionable": [],
                "mset": [],
                "sset": [],
                "activatable": [],
                "to_bp": False,
                "to_ep": False,
                "shuffle": False,
            }
        )
        *_, feats = decode_actions(self._action_tensor(mapper))
        # First action = normal summon, category 0
        assert feats[0, 0, 1].item() == pytest.approx(0 / 10.0)
        # Second action = special summon, category 1
        assert feats[0, 1, 1].item() == pytest.approx(1 / 10.0)

    def test_location_and_sequence(self):
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_CARD,
                "player": 0,
                "cancelable": 0,
                "min": 1,
                "max": 1,
                "cards": [
                    {
                        "code": 100,
                        "controller": 0,
                        "location": LOCATION_MZONE,
                        "sequence": 5,
                        "subsequence": 0,
                    }
                ],
            }
        )
        *_, feats = decode_actions(self._action_tensor(mapper))
        # New layout: feats[..., 3:10] = location bits (after msg_type, category, controller)
        loc_feats = feats[0, 0, 3:10]
        expected_set = _bit_index(_LOC_BITS, LOCATION_MZONE)
        for i in range(7):
            expected = 1.0 if i in expected_set else 0.0
            assert loc_feats[i].item() == expected, f"action loc bit {i}"
        # sequence (heuristic /60.0)
        assert feats[0, 0, 10].item() == pytest.approx(5 / 60.0)

    def test_unused_slots_zero(self):
        """Unused action slots should decode as all zeros."""
        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_CARD,
                "player": 0,
                "cancelable": 0,
                "min": 1,
                "max": 1,
                "cards": [
                    {"code": 100, "controller": 0, "location": 2, "sequence": 0, "subsequence": 0}
                ],
            }
        )
        codes, _, _, feats = decode_actions(self._action_tensor(mapper))
        # Slot 1 (unused) should be all zeros
        assert codes[0, 1].item() == 0
        assert feats[0, 1].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Decoder contract tests (Task 11): pin the 4-tuple shape and the
# disambiguation rule that masks the per-card desc_n scalar to 0 when
# the action carries a sysstring desc.
# ---------------------------------------------------------------------------


class TestDecodeActionsContract:
    """The shape of decode_actions's return is part of the network's
    forward-pass contract; pin it here."""

    def _action_tensor(self, mapper: ActionMapper) -> torch.Tensor:
        return torch.from_numpy(action_features(mapper)).unsqueeze(0)

    def _yesno_tensor(self, desc: int) -> torch.Tensor:
        """Build an action tensor for SELECT_YESNO with a custom desc."""
        from yugioh_core.constants import MSG_SELECT_YESNO

        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_YESNO,
                "player": 0,
                "_agent_player": 0,
                "desc": desc,
            }
        )
        return self._action_tensor(mapper)

    def test_decode_actions_returns_4tuple(self):
        """Pin the (codes, desc_passcodes, desc_ns, action_feats) signature
        so accidental return-shape changes get caught here, not deep in the
        network forward pass."""
        from yugioh_core.constants import MSG_SELECT_YESNO

        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_YESNO,
                "player": 0,
                "_agent_player": 0,
                "desc": 0x1234567890ABC,
            }
        )
        result = decode_actions(self._action_tensor(mapper))
        assert isinstance(result, tuple)
        assert len(result) == 4
        codes, desc_pcs, desc_ns, feats = result
        assert codes.shape == (1, MAX_ACTIONS)
        assert desc_pcs.shape == (1, MAX_ACTIONS)
        assert desc_ns.shape == (1, MAX_ACTIONS)
        assert feats.dim() == 3

    def test_decode_actions_action_feat_dim(self):
        """ACTION_FEAT_DIM is part of the network input contract; pin it."""
        from yugioh_core.constants import MSG_SELECT_YESNO

        mapper = ActionMapper()
        mapper.update(
            {
                "msg_type": MSG_SELECT_YESNO,
                "player": 0,
                "_agent_player": 0,
                "desc": 0,
            }
        )
        *_, feats = decode_actions(self._action_tensor(mapper))
        assert feats.shape[-1] == 23
        assert feats.shape[-1] == ACTION_FEAT_DIM  # constant tracks reality

    def test_decode_action_per_card_desc_n_scalar_masked_when_sysstring(self):
        """The disambiguation rule: per-card desc_n scalar (last dim of
        action_feats) must be 0 for sysstring actions (passcode==0), and
        non-zero for per-card actions (passcode>0). This is the load-bearing
        invariant that lets the MLP cleanly disambiguate the two desc paths
        without learning to mask anything itself."""
        # Sysstring case: passcode=0 (low 20 bits = 70 means !system 70)
        sys_tensor = self._yesno_tensor(desc=70)
        codes_sys, pcs_sys, ns_sys, feats_sys = decode_actions(sys_tensor)
        # Both yesno actions (yes + no) carry the same desc → both passcode=0
        assert pcs_sys[0, 0].item() == 0
        assert ns_sys[0, 0].item() == 70
        # Per-card desc_n scalar (last dim) must be masked to 0 for sysstring
        assert feats_sys[0, 0, -1].item() == 0.0

        # Per-card case: passcode=12345, n=3 → desc = (12345 << 20) | 3
        per_card_desc = (12345 << 20) | 3
        per_card_tensor = self._yesno_tensor(desc=per_card_desc)
        codes_pc, pcs_pc, ns_pc, feats_pc = decode_actions(per_card_tensor)
        assert pcs_pc[0, 0].item() == 12345
        assert ns_pc[0, 0].item() == 3
        # Per-card desc_n scalar must carry the n value, normalized by vocab-1
        from yugioh_core.encoding import PER_CARD_DESC_N_VOCAB

        assert feats_pc[0, 0, -1].item() == pytest.approx(3.0 / (PER_CARD_DESC_N_VOCAB - 1))
