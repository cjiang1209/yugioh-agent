"""Tests for yugioh_env.replay — GameRecording, RecordingOpponent, RecordingEnvironment."""

from __future__ import annotations

import pytest

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
        wrapper = RecordingOpponent(inner, rec)

        msg = {"msg_type": 1, "player": 1}
        action = wrapper.select_action(msg, num_actions=3)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_records_action(self):
        """Each select_action call appends an entry with correct player and msg_type."""
        inner = RandomOpponent(seed=42)
        rec = GameRecording(setup={})
        wrapper = RecordingOpponent(inner, rec)

        msg1 = {"msg_type": 10, "player": 1}
        wrapper.select_action(msg1, num_actions=5)
        msg2 = {"msg_type": 20, "player": 1}
        wrapper.select_action(msg2, num_actions=3)

        assert len(rec.actions) == 2
        assert rec.actions[0]["player"] == 1
        assert rec.actions[0]["msg_type"] == 10
        assert rec.actions[1]["player"] == 1
        assert rec.actions[1]["msg_type"] == 20

    def test_reseed_delegates(self):
        """reseed should not raise."""
        inner = RandomOpponent(seed=1)
        rec = GameRecording(setup={})
        wrapper = RecordingOpponent(inner, rec)
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
            {"msg_type": 16, "player": 1, "action": 5, "num_actions": 8},
            {"msg_type": 16, "player": 1, "action": 3, "num_actions": 4},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)

        a0 = opp.select_action({"msg_type": 16, "player": 1}, num_actions=8)
        assert a0 == 5
        a1 = opp.select_action({"msg_type": 16, "player": 1}, num_actions=4)
        assert a1 == 3

    def test_drift_on_wrong_msg_type(self):
        actions = [
            {"msg_type": 16, "player": 1, "action": 0, "num_actions": 3},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)

        with pytest.raises(RuntimeError, match="drift"):
            opp.select_action({"msg_type": 999, "player": 1}, num_actions=3)

    def test_exhausted_raises(self):
        actions = [
            {"msg_type": 16, "player": 1, "action": 0, "num_actions": 1},
        ]
        cursor = ReplayCursor(actions, agent_player=0)
        opp = ScriptedOpponent(cursor)
        opp.select_action({"msg_type": 16, "player": 1}, num_actions=1)

        with pytest.raises(RuntimeError, match="exhausted"):
            opp.select_action({"msg_type": 16, "player": 1}, num_actions=1)


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
