import numpy as np

from yugioh_core.constants import (
    HINT_NUMBER,
    LOCATION_MZONE,
    MSG_ATTACK,
    MSG_CHAINING,
    MSG_HINT,
)
from yugioh_core.encoding import (
    EVENT_ENTRY_FEATURES,
    EVENT_LAYOUT,
    encode_event_entry,
)


def test_chaining_entry_roundtrips_code_and_desc():
    feat = encode_event_entry(
        msg_type=MSG_CHAINING,
        card_code=89631139,
        controller=1,
        turn_player=0,
        phase=4,
        location=LOCATION_MZONE,
        sequence=2,
        desc=0x1234_0000_0000_0005,
        turn_count=7,
    )
    assert feat.dtype == np.uint8
    assert feat.shape == (EVENT_ENTRY_FEATURES,)
    assert EVENT_LAYOUT.read(feat, "msg_type") == MSG_CHAINING  # raw discriminator
    assert EVENT_LAYOUT.read(feat, "controller") == 1  # raw seat
    assert EVENT_LAYOUT.read(feat, "turn_player") == 0  # raw seat
    assert EVENT_LAYOUT.read(feat, "turn_count") == 7
    assert EVENT_LAYOUT.read(feat, "card_code") == 89631139
    assert EVENT_LAYOUT.read(feat, "desc") == 0x1234_0000_0000_0005


def test_attack_entry_carries_target():
    feat = encode_event_entry(
        msg_type=MSG_ATTACK,
        card_code=100,
        controller=0,
        turn_player=0,
        phase=8,
        location=LOCATION_MZONE,
        sequence=1,
        target_code=200,
        target_location=LOCATION_MZONE,
        target_sequence=3,
        turn_count=2,
    )
    assert EVENT_LAYOUT.read(feat, "target_code") == 200
    assert EVENT_LAYOUT.read(feat, "target_location") == LOCATION_MZONE
    assert EVENT_LAYOUT.read(feat, "target_sequence") == 3


def test_hint_fields_grouped_in_payload():
    feat = encode_event_entry(msg_type=MSG_HINT, hint_type=HINT_NUMBER, hint_value=16)
    assert EVENT_LAYOUT.read(feat, "hint_type") == HINT_NUMBER
    assert EVENT_LAYOUT.read(feat, "hint_value") == 16


def test_empty_default_is_zero_msg_type():
    feat = encode_event_entry(msg_type=0)
    assert EVENT_LAYOUT.read(feat, "msg_type") == 0
