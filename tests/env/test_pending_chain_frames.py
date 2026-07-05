from yugioh_core.constants import MSG_CHAINING
from yugioh_env.server.yugioh_environment import _raw_pending_chain


def test_raw_pending_chain_from_chunk_messages():
    events = [
        {
            "msg_type": MSG_CHAINING,
            "code": 111,
            "controller": 1,
            "location": 8,
            "sequence": 0,
            "desc": 5,
            "chain_link": 1,
        },
        {"msg_type": 99, "foo": 1},  # unrelated msg ignored
        {
            "msg_type": MSG_CHAINING,
            "code": 222,
            "controller": 0,
            "location": 4,
            "sequence": 1,
            "desc": 0,
            "chain_link": 2,
        },
    ]
    out = _raw_pending_chain(events, agent_player=0)
    assert out == [
        {"chain_link": 1, "code": 111, "desc": 5, "controller": 1},  # raw 1, agent 0 -> opp
        {"chain_link": 2, "code": 222, "desc": 0, "controller": 0},  # raw 0, agent 0 -> you
    ]


def test_raw_pending_chain_empty_when_no_chaining():
    assert _raw_pending_chain([{"msg_type": 99}], agent_player=0) == []
