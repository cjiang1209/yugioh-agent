import numpy as np
import torch

from yugioh_core.constants import MSG_CHAINING
from yugioh_core.encoding import EVENT_ENTRY_FEATURES, MAX_EVENT_HISTORY, encode_event_entry
from yugioh_rl.features import EVENT_FEAT_DIM, decode_event_history


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
    assert aux.shape[-1] == 5
    assert feats.shape[-1] == EVENT_FEAT_DIM


def test_turn_delta_relative_to_newest():
    r_old = encode_event_entry(msg_type=MSG_CHAINING, card_code=1, turn_count=2)
    r_new = encode_event_entry(msg_type=MSG_CHAINING, card_code=2, turn_count=6)
    raw = np.zeros((1, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
    raw[0, 30] = r_old
    raw[0, 31] = r_new
    _, _, _, _, _, feats = decode_event_history(torch.from_numpy(raw))
    # turn_delta is the LAST feature column; newest row delta 0, older row delta 4
    assert feats[0, 31, -1].item() == 0.0
    assert feats[0, 30, -1].item() == 4.0
