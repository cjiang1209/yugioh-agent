from core.constants import (
    HINT_CODE,
    LOCATION_MZONE,
    MSG_HINT,
    MSG_SUMMONING,
    PHASE_END,
    PHASE_MAIN1,
    PHASE_MAIN2,
)
from core.encoding import (
    EVENT_ENTRY_FEATURES,
    EVENT_LAYOUT,
    MAX_EVENT_HISTORY,
)
from env.event_buffer import EventHistoryBuffer


def _summon(code, controller, seq):
    return {
        "msg_type": MSG_SUMMONING,
        "code": code,
        "controller": controller,
        "location": LOCATION_MZONE,
        "sequence": seq,
    }


def test_append_and_right_aligned_tensor():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 0, 1)], turn_count=1, current_player=0, phase=4)
    b.append_from_enriched([_summon(222, 1, 2)], turn_count=2, current_player=1, phase=4)
    t = b.to_tensor(agent_player=0)
    assert t.shape == (MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    # newest (222) at row 31, older (111) at row 30, rest empty (msg_type byte 0 == 0)
    assert t[31, EVENT_LAYOUT.offsets["msg_type"]] == MSG_SUMMONING
    assert t[30, EVENT_LAYOUT.offsets["msg_type"]] == MSG_SUMMONING
    assert t[29, EVENT_LAYOUT.offsets["msg_type"]] == 0
    assert EVENT_LAYOUT.read(t[31], "card_code") == 222


def test_controller_relativized_at_encode():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 1, 1)], turn_count=1, current_player=1, phase=4)
    # agent_player=1 → raw controller 1 becomes relative 0 (=me); controller=[1], turn_player=[2]
    t = b.to_tensor(agent_player=1)
    assert t[31, EVENT_LAYOUT.offsets["controller"]] == 0
    assert t[31, EVENT_LAYOUT.offsets["turn_player"]] == 0  # turn_player raw 1, agent 1 → 0
    # agent_player=0 → raw controller 1 becomes relative 1 (=opp)
    t0 = b.to_tensor(agent_player=0)
    assert t0[31, EVENT_LAYOUT.offsets["controller"]] == 1
    assert t0[31, EVENT_LAYOUT.offsets["turn_player"]] == 1


def test_hint_code_records_declared_passcode():
    b = EventHistoryBuffer()
    b.append_from_enriched(
        [{"msg_type": MSG_HINT, "hint_type": HINT_CODE, "player": 0, "data": 55144522}],
        turn_count=3,
        current_player=0,
        phase=4,
    )
    t = b.to_tensor(agent_player=0)
    assert t[31, EVENT_LAYOUT.offsets["msg_type"]] == MSG_HINT
    assert t[31, EVENT_LAYOUT.offsets["hint_type"]] == HINT_CODE
    assert EVENT_LAYOUT.read(t[31], "card_code") == 55144522


def test_non_declaration_hint_ignored():
    b = EventHistoryBuffer()
    b.append_from_enriched(
        [{"msg_type": MSG_HINT, "hint_type": 3, "player": 0, "data": 1}],
        turn_count=1,
        current_player=0,
        phase=4,
    )
    t = b.to_tensor(agent_player=0)
    assert t[31, EVENT_LAYOUT.offsets["msg_type"]] == 0  # nothing recorded


def test_reset_clears():
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(111, 0, 1)], turn_count=1, current_player=0, phase=4)
    b.reset()
    t = b.to_tensor(agent_player=0)
    assert t[31, EVENT_LAYOUT.offsets["msg_type"]] == 0


def test_maxlen_evicts_oldest():
    b = EventHistoryBuffer()
    for i in range(40):
        b.append_from_enriched([_summon(1000 + i, 0, 1)], turn_count=1, current_player=0, phase=4)
    t = b.to_tensor(agent_player=0)
    # 32 kept; newest (1039) at row 31
    assert EVENT_LAYOUT.read(t[31], "card_code") == 1039
    assert all(t[r, EVENT_LAYOUT.offsets["msg_type"]] != 0 for r in range(32))  # full


def test_phase_stored_as_bit_index():
    # Phase is a single-bit flag; the buffer stores the compact bit-index so
    # high phases (MAIN2=0x100, END=0x200) survive the 1-byte feature slot.
    b = EventHistoryBuffer()
    b.append_from_enriched([_summon(1, 0, 1)], turn_count=1, current_player=0, phase=PHASE_MAIN1)
    b.append_from_enriched([_summon(2, 0, 1)], turn_count=1, current_player=0, phase=PHASE_MAIN2)
    b.append_from_enriched([_summon(3, 0, 1)], turn_count=1, current_player=0, phase=PHASE_END)
    t = b.to_tensor(agent_player=0)
    assert t[29, EVENT_LAYOUT.offsets["phase"]] == 2  # MAIN1 (0x04) → bit 2
    assert t[30, EVENT_LAYOUT.offsets["phase"]] == 8  # MAIN2 (0x100) → bit 8
    assert t[31, EVENT_LAYOUT.offsets["phase"]] == 9  # END  (0x200) → bit 9
