from unittest.mock import patch

import pytest

from yugioh_core.constants import LOCATION_MZONE
from yugioh_env.server.serving_env import (
    FrameCollector,
    ServingEnv,
    capture_board,
    capture_game_state,
)

_ENV_CLASS = "yugioh_env.server.yugioh_environment.YuGiOhEnvironment"


@pytest.fixture
def env(lib, db_path, script_dirs, deck_path):
    """Create a real YuGiOhEnvironment instance for observer-seam integration tests."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    config = {
        "lib_path": None,  # auto-detect
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
    }
    e = YuGiOhEnvironment(config)
    yield e
    e.close()


class _StubGS:
    turn_count = 3
    phase = 1
    current_player = 0
    chain_count = 0
    pending_chain = []
    lp = {0: 8000, 1: 7000}
    deck_count = {0: 30, 1: 31}
    hand_count = {0: 5, 1: 4}
    extra_count = {0: 2, 1: 1}


class _StubView:
    def __init__(self, gs=None, cards=None, agent_player=0):
        self._gs = gs or _StubGS()
        self._cards = cards or {}
        self.agent_player = agent_player
        self.query_calls = 0

    @property
    def game_state(self):
        return self._gs

    def query_location(self, p, loc):
        self.query_calls += 1
        return self._cards.get((p, loc), [])


def test_capture_board_carries_dynamic_fields_unhidden():
    card = {
        "code": 111,
        "position": 1,
        "sequence": 0,
        "type": 0x1,
        "attack": 4000,
        "defense": 3000,
        "level": 8,
    }
    raw = capture_board(_StubView(cards={(0, LOCATION_MZONE): [card]}))
    got = raw["agent"]["monsters"][0]
    assert got["attack"] == 4000 and got["code"] == 111
    assert raw["agent"]["lp"] == 8000 and raw["opponent"]["lp"] == 7000
    assert raw["agent"]["deck_count"] == 30 and raw["agent_player"] == 0


def test_capture_game_state_does_no_zone_queries():
    view = _StubView()
    gs = capture_game_state(view)
    assert view.query_calls == 0  # perf guard: no FFI for top-level game_state
    assert gs == {"turn": 3, "phase": 1, "is_my_turn": True, "chain_count": 0, "pending_chain": []}


def test_capture_no_duel_returns_empty():
    class _NoDuel:
        agent_player = 0
        game_state = None

        def query_location(self, p, loc):
            return []

    v = _NoDuel()
    assert capture_board(v) == {"agent": {}, "opponent": {}}
    assert capture_game_state(v) == {
        "turn": 0,
        "phase": None,
        "is_my_turn": False,
        "chain_count": 0,
        "pending_chain": [],
    }


def test_collector_lifecycle():
    c = FrameCollector()
    c.begin()
    c.on_chunk([{"msg_type": 1}], _StubView())
    frames = c.take()
    assert len(frames) == 1 and frames[0]["events"] == [{"msg_type": 1}]
    assert c.take() == []  # drained


def test_pending_chain_from_live_game_state_when_no_events():
    """Non-identity: agent_player=1, engine controller 0 → relative 1."""

    class _Link:
        chain_link, code, desc, controller = 1, 555, 7, 0

    class _GS(_StubGS):
        pending_chain = [_Link()]

    gs = capture_game_state(_StubView(gs=_GS(), agent_player=1))
    (e,) = gs["pending_chain"]
    assert e["code"] == 555 and e["chain_link"] == 1 and e["controller"] == 1


def test_pending_chain_from_chunk_events_overrides_live():
    from yugioh_core.constants import MSG_CHAINING

    class _Link:
        chain_link, code, desc, controller = 9, 999, 0, 0  # live — ignored

    class _GS(_StubGS):
        pending_chain = [_Link()]

    events = [{"msg_type": MSG_CHAINING, "chain_link": 1, "code": 777, "desc": 3, "controller": 0}]
    gs = capture_game_state(_StubView(gs=_GS()), events=events)
    (e,) = gs["pending_chain"]
    assert e["code"] == 777 and e["controller"] == 0  # chunk overrides; agent 0 == engine 0


class _MiniView:
    agent_player = 0
    game_state = None

    def query_location(self, p, loc):
        return []


class _FakeCore:
    def __init__(self):
        self._observer = None

    def set_frame_observer(self, o):
        self._observer = o

    def step(self, action, **kw):
        self._observer.on_chunk([{"msg_type": 1}], _MiniView())
        return "STEP_OBS"


def test_serving_env_step_returns_frames():
    with patch(_ENV_CLASS, return_value=_FakeCore()):
        s = ServingEnv()
        obs, frames = s.step(action=object())
        assert obs == "STEP_OBS" and len(frames) == 1


def test_serving_frames_capture_per_chunk(env):
    from yugioh_env.replay import GameRecording, ScriptedOpponent

    recording = GameRecording(
        setup={
            "puzzle": {
                "player0": {"hand": [69140098], "deck": [89631139, 89631139]},
                "player1": {"hand": [89631139], "deck": [89631139]},
            },
            "agent_player": 1,
        }
    )
    recording.append(msg_type=11, player=0, action=0, num_actions=3)
    recording.append(msg_type=18, player=0, action=0, num_actions=5)
    recording.append(msg_type=16, player=1, action=0, num_actions=1)
    with patch(_ENV_CLASS, return_value=env):
        serving = ServingEnv()
    serving.env.set_opponent(ScriptedOpponent(recording.cursor()))
    _obs, frames = serving.reset(puzzle=recording.setup["puzzle"], agent_player=1)
    assert len(frames) == 3

    def occ(side):
        return sum(1 for m in side if m.get("code"))

    assert occ(frames[-1]["board"]["opponent"]["monsters"]) > occ(
        frames[0]["board"]["opponent"]["monsters"]
    )


def test_serving_multi_select_substep_returns_no_frames(env):
    from yugioh_core.constants import MSG_SELECT_IDLECMD, MSG_SELECT_TRIBUTE
    from yugioh_env.models import YuGiOhAction

    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": 89631139, "pos": "face_up_attack", "seq": 0},
                {"code": 89631139, "pos": "face_up_attack", "seq": 1},
            ],
            "hand": [46986414],
            "deck": [89631139],
        }
    }
    with patch(_ENV_CLASS, return_value=env):
        serving = ServingEnv()
    serving.reset(puzzle=puzzle)
    assert serving.env._mapper.msg_type == MSG_SELECT_IDLECMD
    serving.step(YuGiOhAction(action_index=0))
    assert serving.env._mapper.msg_type == MSG_SELECT_TRIBUTE
    _obs, frames = serving.step(YuGiOhAction(action_index=0))
    assert frames == []
