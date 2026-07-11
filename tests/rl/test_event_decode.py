import numpy as np
import torch

from yugioh_core.constants import MSG_CHAINING
from yugioh_core.encoding import EVENT_ENTRY_FEATURES, MAX_EVENT_HISTORY, encode_event_entry
from yugioh_rl.features import EVENT_FEAT_DIM, decode_event_history

# feats column layout (see decode_event_history): scalars first, then blocks.
# [controller, turn_player, sequence, target_sequence, hint_value, turn_delta,
#  location(7), target_location(7), hint_onehot(4)]
_TURN_DELTA_COL = 5


def _raw_one(entry_row):
    raw = np.zeros((1, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
    raw[0, MAX_EVENT_HISTORY - 1] = entry_row
    return torch.from_numpy(raw)


def test_decode_desc_split_matches_chain_convention():
    # desc_n (low 20 bits) must fit SYSSTRING_VOCAB (65536); use 0x0ABCD.
    row = encode_event_entry(
        msg_type=MSG_CHAINING, card_code=111, desc=(0xABCDE << 20) | 0x0ABCD, turn_count=5
    )
    codes, desc_pass, desc_ns, tgt, aux, feats = decode_event_history(_raw_one(row))
    assert codes[0, -1].item() == 111
    assert desc_pass[0, -1].item() == 0xABCDE
    assert desc_ns[0, -1].item() == 0x0ABCD
    assert aux.shape[-1] == 2  # msg_type, phase (only embedded categoricals)
    assert feats.shape[-1] == EVENT_FEAT_DIM


def test_turn_delta_relative_to_newest():
    r_old = encode_event_entry(msg_type=MSG_CHAINING, card_code=1, turn_count=2)
    r_new = encode_event_entry(msg_type=MSG_CHAINING, card_code=2, turn_count=6)
    raw = np.zeros((1, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
    raw[0, 30] = r_old
    raw[0, 31] = r_new
    _, _, _, _, _, feats = decode_event_history(torch.from_numpy(raw))
    # newest row delta 0, older row delta 4
    assert feats[0, 31, _TURN_DELTA_COL].item() == 0.0
    assert feats[0, 30, _TURN_DELTA_COL].item() == 4.0


def test_location_is_onehot_not_scalar():
    # location byte is a zone bitmask; decode must bit-expand it (grave=0x10),
    # not feed the raw value 16 as a scalar. A card is in exactly one zone, so
    # the expansion is one-hot.
    from yugioh_core.constants import MSG_SUMMONING

    # feats layout: [ctrl, turn, seq, tgt_seq, hint_value, turn_delta,
    #                location(7), target_location(7), hint_onehot(4)]
    loc_start = 6
    # _LOC_BITS order: hand,mzone,szone,grave,banished,extra,deck → grave at idx 3
    row = encode_event_entry(msg_type=MSG_SUMMONING, card_code=1, location=0x10, sequence=2)
    _, _, _, _, _, feats = decode_event_history(_raw_one(row))
    loc = feats[0, -1, loc_start : loc_start + 7]
    assert loc[3].item() == 1.0  # grave bit set
    assert loc.sum().item() == 1.0  # exactly one zone, not a scalar 16


def test_controller_turn_player_are_scalars():
    from yugioh_core.constants import MSG_SUMMONING

    row = encode_event_entry(msg_type=MSG_SUMMONING, card_code=1, controller=1, turn_player=0)
    _, _, _, _, _, feats = decode_event_history(_raw_one(row))
    assert feats[0, -1, 0].item() == 1.0  # controller scalar
    assert feats[0, -1, 1].item() == 0.0  # turn_player scalar


def test_hint_type_is_onehot():
    from yugioh_core.constants import HINT_ATTRIB, MSG_HINT

    # hint one-hot occupies the last 4 columns; _EVENT_HINT_IDS order
    # [race, attrib, code, number] → ATTRIB at idx 1.
    row = encode_event_entry(msg_type=MSG_HINT, hint_type=HINT_ATTRIB, hint_value=0x40)
    _, _, _, _, _, feats = decode_event_history(_raw_one(row))
    hint = feats[0, -1, -4:]
    assert hint[1].item() == 1.0  # attrib column set
    assert hint.sum().item() == 1.0  # exactly one hint kind
