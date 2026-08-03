"""Tests for yugioh_env.replay — GameRecording, RecordingOpponent, RecordingEnvironment."""

from __future__ import annotations

import pytest

from tests.env.conftest import MINIMAL_MSGS, obs_from_msg
from yugioh_core.constants import MSG_SELECT_BATTLECMD, MSG_SELECT_IDLECMD, MSG_SELECT_YESNO
from yugioh_env.opponent import GreedyOpponent, RandomOpponent
from yugioh_env.replay import (
    GameRecording,
    RecordingEnvironment,
    RecordingOpponent,
    ReplayCursor,
    ScriptedOpponent,
)

# ---------------------------------------------------------------------------
# Unit tests — no engine required
# ---------------------------------------------------------------------------


class TestGameRecording:
    def test_empty_recording(self):
        rec = GameRecording(setup={"seed": 1})
        assert rec.setup == {"seed": 1}
        assert rec.actions == []

    def test_append_action(self):
        rec = GameRecording(setup={"seed": 1})
        rec.append(msg_type=10, player=0, action=3, num_actions=5)
        rec.append(msg_type=20, player=1, action=0, num_actions=2)
        assert len(rec.actions) == 2
        assert rec.actions[0] == {
            "msg_type": 10,
            "player": 0,
            "action": 3,
            "num_actions": 5,
        }
        assert rec.actions[1] == {
            "msg_type": 20,
            "player": 1,
            "action": 0,
            "num_actions": 2,
        }

    def test_save_and_load_json(self, tmp_path):
        rec = GameRecording(setup={"seed": 42, "agent_player": 0})
        rec.append(msg_type=10, player=0, action=1, num_actions=3)
        rec.append(msg_type=20, player=1, action=0, num_actions=2)

        path = tmp_path / "recording.json"
        rec.save(path)

        loaded = GameRecording.load(path)
        assert loaded.setup == rec.setup
        assert loaded.actions == rec.actions

    def test_save_and_load_puzzle_setup(self, tmp_path):
        setup = {
            "seed": 1,
            "puzzle": {
                "players": [
                    {"lp": 8000, "hand": [89631139]},
                    {"lp": 4000, "hand": []},
                ],
            },
        }
        rec = GameRecording(setup=setup)
        rec.append(msg_type=5, player=0, action=0, num_actions=1)

        path = tmp_path / "puzzle_rec.json"
        rec.save(path)

        loaded = GameRecording.load(path)
        assert loaded.setup["puzzle"] == setup["puzzle"]
        assert loaded.actions == rec.actions


class TestRecordingOpponent:
    def test_delegates_to_inner(self):
        """Wrapped opponent returns a valid action index."""
        inner = GreedyOpponent()
        rec = GameRecording(setup={})
        wrapper = RecordingOpponent(inner, rec, seat_fn=lambda: 1)

        obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_IDLECMD], "msg_type": MSG_SELECT_IDLECMD})
        num_actions = int(obs.action_mask.sum())
        action = wrapper.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < num_actions

    def test_records_action(self):
        """Each select_action call appends an entry with correct player and msg_type."""
        inner = RandomOpponent(seed=42)
        rec = GameRecording(setup={})
        wrapper = RecordingOpponent(inner, rec, seat_fn=lambda: 1)

        obs1 = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_IDLECMD], "msg_type": MSG_SELECT_IDLECMD})
        wrapper.select_action(obs1)
        obs2 = obs_from_msg(
            {**MINIMAL_MSGS[MSG_SELECT_BATTLECMD], "msg_type": MSG_SELECT_BATTLECMD}
        )
        wrapper.select_action(obs2)

        assert len(rec.actions) == 2
        assert rec.actions[0]["player"] == 1
        assert rec.actions[0]["msg_type"] == MSG_SELECT_IDLECMD
        assert rec.actions[1]["player"] == 1
        assert rec.actions[1]["msg_type"] == MSG_SELECT_BATTLECMD

    def test_reseed_delegates(self):
        """reseed should not raise."""
        inner = RandomOpponent(seed=1)
        rec = GameRecording(setup={})
        wrapper = RecordingOpponent(inner, rec, seat_fn=lambda: 1)
        wrapper.reseed(99)  # should not raise


# ---------------------------------------------------------------------------
# Integration tests — require engine
# ---------------------------------------------------------------------------


def _make_recording_env(db_path, script_dirs, deck_path):
    """Create a YuGiOhEnvironment + RecordingEnvironment for testing."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
    }
    env = YuGiOhEnvironment(config)
    opponent = RandomOpponent(seed=99)
    recorder = RecordingEnvironment(env, opponent)
    return recorder, env


class TestRecordingEnvironment:
    @pytest.fixture
    def rec_env(self, lib, db_path, script_dirs, deck_path):
        recorder, env = _make_recording_env(db_path, script_dirs, deck_path)
        yield recorder
        env.close()

    def test_records_agent_actions(self, rec_env):
        """Agent actions should be recorded with valid fields."""
        obs = rec_env.reset(seed=42)
        steps = 0
        while not obs.done and steps < 5:
            obs = rec_env.step(action_index=0)
            steps += 1

        recording = rec_env.get_recording()
        assert len(recording.actions) > 0

        for entry in recording.actions:
            assert "msg_type" in entry
            assert "player" in entry
            assert "action" in entry
            assert "num_actions" in entry
            assert entry["player"] in (0, 1)
            assert entry["action"] >= 0
            assert entry["num_actions"] > 0

    def test_recording_has_both_players(self, rec_env):
        """A full game recording should contain entries for both players."""
        obs = rec_env.reset(seed=42)
        while not obs.done:
            obs = rec_env.step(action_index=0)

        recording = rec_env.get_recording()
        players_seen = {entry["player"] for entry in recording.actions}
        assert 0 in players_seen, "No entries for player 0"
        assert 1 in players_seen, "No entries for player 1"

    def test_get_recording_raises_before_reset(self):
        """get_recording should raise if reset hasn't been called."""
        rec_env = RecordingEnvironment(env=None, opponent=RandomOpponent(seed=1))
        with pytest.raises(RuntimeError, match="No active recording"):
            rec_env.get_recording()

    def test_recording_stamps_correct_seat_when_agent_player_is_random(self, rec_env):
        """Opponent may act during reset(), before the caller can read the seat."""
        rec_env.reset(agent_player="random", seed=7)
        resolved = rec_env._env._agent_player

        # Recording entries are DICTS -- {"msg_type", "player", ...}
        #
        # Do NOT filter by `e["player"] != resolved` first: that discards
        # exactly the entries a broken seat_fn mis-stamps as the agent, so
        # the assertion would inspect only the already-correct ones. During
        # reset() the agent has not acted, so EVERY recorded entry must be
        # the opponent's.
        entries = rec_env._recording.actions
        assert entries, "opponent must have acted during reset"
        assert all(e["player"] == 1 - resolved for e in entries), (
            "every reset-time entry must carry the resolved opponent seat"
        )


# ---------------------------------------------------------------------------
# Unit tests — ReplayCursor
# ---------------------------------------------------------------------------

CURSOR_ACTIONS = [
    {"msg_type": 11, "player": 0, "action": 2, "num_actions": 8},
    {"msg_type": 16, "player": 1, "action": 0, "num_actions": 3},
    {"msg_type": 10, "player": 0, "action": 1, "num_actions": 5},
    {"msg_type": 16, "player": 1, "action": 1, "num_actions": 2},
]


class TestReplayCursor:
    def test_next_agent_entry(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        entry = cursor.next_agent_entry()
        assert entry["player"] == 0
        assert entry["msg_type"] == 11
        assert entry["action"] == 2

    def test_next_opponent_entry(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        # Skip agent entry first
        cursor.next_agent_entry()
        entry = cursor.next_opponent_entry()
        assert entry["player"] == 1
        assert entry["msg_type"] == 16
        assert entry["action"] == 0

    def test_interleaved_order(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        e0 = cursor.next_agent_entry()
        e1 = cursor.next_opponent_entry()
        e2 = cursor.next_agent_entry()
        e3 = cursor.next_opponent_entry()
        assert [e["player"] for e in [e0, e1, e2, e3]] == [0, 1, 0, 1]
        assert cursor.exhausted

    def test_drift_wrong_player(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        # First entry is player 0 (agent), calling next_opponent_entry should drift
        with pytest.raises(RuntimeError, match="drift"):
            cursor.next_opponent_entry()

    def test_exhausted_raises(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        cursor.next_agent_entry()
        cursor.next_opponent_entry()
        cursor.next_agent_entry()
        cursor.next_opponent_entry()
        assert cursor.exhausted
        with pytest.raises(RuntimeError, match="exhausted"):
            cursor.next_agent_entry()

    def test_msg_type_verified(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        # Should not raise — msg_type matches
        cursor.next_agent_entry(expected_msg_type=11)

    def test_msg_type_drift(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        with pytest.raises(RuntimeError, match="drift"):
            cursor.next_agent_entry(expected_msg_type=999)

    def test_num_actions_drift(self):
        cursor = ReplayCursor(CURSOR_ACTIONS, agent_player=0)
        with pytest.raises(RuntimeError, match="drift"):
            cursor.next_agent_entry(expected_num_actions=999)


# ---------------------------------------------------------------------------
# Unit tests — ScriptedOpponent
# ---------------------------------------------------------------------------


class TestScriptedOpponent:
    def test_returns_recorded_actions(self):
        # Two opponent entries (player=1), agent_player=0
        actions = [
            {"msg_type": MSG_SELECT_YESNO, "player": 1, "action": 1, "num_actions": 2},
            {"msg_type": MSG_SELECT_YESNO, "player": 1, "action": 0, "num_actions": 2},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)

        obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_YESNO], "msg_type": MSG_SELECT_YESNO})
        a0 = opp.select_action(obs)
        assert a0 == 1
        a1 = opp.select_action(obs)
        assert a1 == 0

    def test_drift_on_wrong_msg_type(self):
        actions = [
            {"msg_type": 999, "player": 1, "action": 0, "num_actions": 2},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)

        obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_YESNO], "msg_type": MSG_SELECT_YESNO})
        with pytest.raises(RuntimeError, match="drift"):
            opp.select_action(obs)

    def test_drift_on_wrong_num_actions(self):
        """Sibling of test_drift_on_wrong_msg_type: msg_type matches but the
        recorded num_actions doesn't match the live observation's legal-action
        count. ScriptedOpponent passes expected_num_actions on every call
        (see select_action); a recording/replay desync in the action space
        (e.g. a legal-action-count mismatch from a filter change) must not
        pass silently."""
        actions = [
            {"msg_type": MSG_SELECT_YESNO, "player": 1, "action": 0, "num_actions": 999},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)

        obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_YESNO], "msg_type": MSG_SELECT_YESNO})
        with pytest.raises(RuntimeError, match="drift"):
            opp.select_action(obs)

    def test_exhausted_raises(self):
        actions = [
            {"msg_type": MSG_SELECT_YESNO, "player": 1, "action": 0, "num_actions": 2},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)
        obs = obs_from_msg({**MINIMAL_MSGS[MSG_SELECT_YESNO], "msg_type": MSG_SELECT_YESNO})
        opp.select_action(obs)

        with pytest.raises(RuntimeError, match="exhausted"):
            opp.select_action(obs)


# ---------------------------------------------------------------------------
# Integration tests — record then replay
# ---------------------------------------------------------------------------


class TestRecordAndReplay:
    @pytest.fixture
    def rec_env(self, lib, db_path, script_dirs, deck_path):
        recorder, env = _make_recording_env(db_path, script_dirs, deck_path)
        yield recorder, env, str(deck_path)
        env.close()

    def _record_game(self, rec_env, seed=42, agent_player=0):
        """Record a full game, always picking action 0. Returns (recording, final_reward)."""
        recorder, _, _ = rec_env
        obs = recorder.reset(seed=seed, agent_player=agent_player)
        while not obs.done:
            obs = recorder.step(action_index=0)
        return recorder.get_recording(), obs.reward

    def _replay_game(self, lib, db_path, script_dirs, deck_path, recording):
        """Replay a recording, driving agent actions from the cursor."""
        from yugioh_env.models import YuGiOhAction
        from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

        cursor = recording.cursor()
        scripted = ScriptedOpponent(cursor)

        config = {
            "db_path": str(db_path),
            "script_dirs": [str(d) for d in script_dirs],
            "deck_path": deck_path,
            "opponent": "random",  # will be overridden
        }
        env = YuGiOhEnvironment(config)
        env.set_opponent(scripted)

        setup = recording.setup
        obs = env.reset(
            seed=setup["seed"],
            agent_player=setup["agent_player"],
        )

        while not obs.done:
            entry = cursor.next_agent_entry(expected_msg_type=env._mapper.msg_type)
            obs = env.step(YuGiOhAction(action_index=entry["action"]))

        result_reward = obs.reward
        env.close()
        return result_reward, cursor

    def test_record_then_replay_same_outcome(self, rec_env, lib, db_path, script_dirs):
        """Record a game, replay it, verify same winner and LP."""
        recording, orig_reward = self._record_game(rec_env)
        _, _, deck_path = rec_env

        replay_reward, cursor = self._replay_game(lib, db_path, script_dirs, deck_path, recording)

        assert replay_reward == orig_reward, f"Reward mismatch: {replay_reward} vs {orig_reward}"
        assert cursor.exhausted

    def test_drift_detected_with_wrong_seed(self, rec_env, lib, db_path, script_dirs):
        """Replaying with a different seed should cause drift."""
        _, _, deck_path = rec_env
        recording, _ = self._record_game(rec_env, seed=42)

        # Tamper with the seed
        recording.setup["seed"] = 999

        with pytest.raises(RuntimeError, match="drift|exhausted"):
            self._replay_game(lib, db_path, script_dirs, deck_path, recording)

    def test_save_load_replay(self, rec_env, lib, db_path, script_dirs, tmp_path):
        """Record, save to JSON, load, replay — same outcome."""
        recording, orig_reward = self._record_game(rec_env)
        _, _, deck_path = rec_env

        path = tmp_path / "test_recording.json"
        recording.save(path)
        loaded = GameRecording.load(path)

        replay_reward, cursor = self._replay_game(lib, db_path, script_dirs, deck_path, loaded)

        assert replay_reward == orig_reward
        assert cursor.exhausted
