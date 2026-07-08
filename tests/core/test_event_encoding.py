import numpy as np

from yugioh_core.constants import MSG_ATTACK, MSG_CHAINING
from yugioh_core.encoding import (
    EVENT_ENTRY_FEATURES,
    MAX_EVENT_HISTORY,
    decode_u32,
    encode_event_entry,
)


def test_constants():
    assert MAX_EVENT_HISTORY == 32
    assert EVENT_ENTRY_FEATURES == 30


def test_chaining_entry_roundtrips_code_and_desc():
    feat = encode_event_entry(
        msg_type=MSG_CHAINING,
        card_code=89631139,
        controller=1,
        turn_player=0,
        phase=4,
        location=0x04,
        sequence=2,
        desc=0x1234_0000_0000_0005,
        turn_count=7,
    )
    assert feat.dtype == np.uint8
    assert feat.shape == (30,)
    assert feat[0] == MSG_CHAINING  # raw msg_type discriminator
    assert feat[1] == 1  # raw controller
    assert feat[2] == 0  # raw turn_player
    assert feat[4] == 7  # turn_count (uint8)
    assert decode_u32(feat, 5) == 89631139  # card_code [5:9]
    desc = int.from_bytes(bytes(feat[17:25]), "little")
    assert desc == 0x1234_0000_0000_0005  # desc [17:25]


def test_attack_entry_carries_target():
    feat = encode_event_entry(
        msg_type=MSG_ATTACK,
        card_code=100,
        controller=0,
        turn_player=0,
        phase=8,
        location=0x04,
        sequence=1,
        target_code=200,
        target_location=0x04,
        target_sequence=3,
        turn_count=2,
    )
    assert decode_u32(feat, 11) == 200  # target_code [11:15]
    assert feat[15] == 0x04  # target_location
    assert feat[16] == 3  # target_sequence


def test_hint_fields_grouped_in_payload():
    feat = encode_event_entry(msg_type=2, hint_type=9, hint_value=16)
    assert feat[25] == 9  # hint_type [25]
    assert decode_u32(feat, 26) == 16  # hint_value [26:30]


def test_empty_default_is_zero_msg_type():
    feat = encode_event_entry(msg_type=0)
    assert feat[0] == 0
