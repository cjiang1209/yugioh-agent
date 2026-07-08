from yugioh_core.constants import (
    HINT_CODE,
    MSG_HINT,
    MSG_SUMMONING,
)
from yugioh_core.encoding import decode_u32
from yugioh_env.event_buffer import EventHistoryBuffer


def _summon(code, controller, seq):
    return {
        "msg_type": MSG_SUMMONING,
        "code": code,
        "controller": controller,
        "location": 0x04,
        "sequence": seq,
    }


def test_append_and_right_aligned_tensor():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 0, 1)], turn_count=1, current_player=0, phase=4)
    b.append_from_enriched([_summon(222, 1, 2)], turn_count=2, current_player=1, phase=4)
    t = b.to_tensor(agent_player=0)
    assert t.shape == (32, 30)
    # newest (222) at row 31, older (111) at row 30, rest empty (msg_type byte 0 == 0)
    assert t[31, 0] == MSG_SUMMONING
    assert t[30, 0] == MSG_SUMMONING
    assert t[29, 0] == 0
    assert decode_u32(t[31], 5) == 222


def test_controller_relativized_at_encode():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 1, 1)], turn_count=1, current_player=1, phase=4)
    # agent_player=1 → raw controller 1 becomes relative 0 (=me); controller=[1], turn_player=[2]
    t = b.to_tensor(agent_player=1)
    assert t[31, 1] == 0
    assert t[31, 2] == 0  # turn_player raw 1, agent 1 → 0
    # agent_player=0 → raw controller 1 becomes relative 1 (=opp)
    t0 = b.to_tensor(agent_player=0)
    assert t0[31, 1] == 1
    assert t0[31, 2] == 1


def test_hint_code_records_declared_passcode():
    b = EventHistoryBuffer()
    b.append_from_enriched(
        [{"msg_type": MSG_HINT, "hint_type": HINT_CODE, "player": 0, "data": 55144522}],
        turn_count=3,
        current_player=0,
        phase=4,
    )
    t = b.to_tensor(agent_player=0)
    assert t[31, 0] == MSG_HINT
    assert t[31, 25] == HINT_CODE  # hint_type at [25]
    assert decode_u32(t[31], 5) == 55144522


def test_non_declaration_hint_ignored():
    b = EventHistoryBuffer()
    b.append_from_enriched(
        [{"msg_type": MSG_HINT, "hint_type": 3, "player": 0, "data": 1}],
        turn_count=1,
        current_player=0,
        phase=4,
    )
    t = b.to_tensor(agent_player=0)
    assert t[31, 0] == 0  # nothing recorded


def test_reset_clears():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 0, 1)], turn_count=1, current_player=0, phase=4)
    b.reset()
    t = b.to_tensor(agent_player=0)
    assert t[31, 0] == 0


def test_maxlen_evicts_oldest():
    b = EventHistoryBuffer()
    for i in range(40):
        b.append_from_enriched([_summon(1000 + i, 0, 1)], turn_count=1, current_player=0, phase=4)
    t = b.to_tensor(agent_player=0)
    # 32 kept; newest (1039) at row 31
    assert decode_u32(t[31], 5) == 1039
    assert all(t[r, 0] != 0 for r in range(32))  # full


def test_phase_stored_as_bit_index():
    # Phase is a single-bit flag; the buffer stores the compact bit-index so
    # high phases (MAIN2=0x100, END=0x200) survive the 1-byte feature slot.
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(1, 0, 1)], turn_count=1, current_player=0, phase=0x04)
    b.append_from_enriched([_summon(2, 0, 1)], turn_count=1, current_player=0, phase=0x100)
    b.append_from_enriched([_summon(3, 0, 1)], turn_count=1, current_player=0, phase=0x200)
    t = b.to_tensor(agent_player=0)
    assert t[29, 3] == 2  # MAIN1 (0x04) → bit 2
    assert t[30, 3] == 8  # MAIN2 (0x100) → bit 8
    assert t[31, 3] == 9  # END  (0x200) → bit 9
