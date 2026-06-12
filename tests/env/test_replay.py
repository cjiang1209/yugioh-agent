"""Tests for yugioh_env.replay — GameRecording, RecordingOpponent, RecordingEnvironment."""

from __future__ import annotations

import pytest

from yugioh_env.opponent import GreedyOpponent, RandomOpponent
from yugioh_env.replay import GameRecording, RecordingEnvironment, RecordingOpponent

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
