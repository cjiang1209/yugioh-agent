"""Unit and integration tests for ActionLoopFilter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from yugioh_core.constants import MSG_SELECT_CHAIN, MSG_SELECT_IDLECMD
from yugioh_env.action_loop_filter import _ZONES, LOOP_DETECTION_THRESHOLD, ActionLoopFilter
from yugioh_env.models import YuGiOhAction
from yugioh_env.puzzle import generate_disable_lua
from yugioh_env.replay import GameRecording, ScriptedOpponent
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

# ─── Generic test actions ────────────────────────────────────────────────────

_ACTION_A: dict = {
    "code": 1001,
    "controller": 0,
    "location": 0x10,
    "sequence": 0,
    "category": 5,
    "desc": 0,
}
_ACTION_B: dict = {
    "code": 1002,
    "controller": 0,
    "location": 0x04,
    "sequence": 1,
    "category": 1,
    "desc": 0,
}
_ACTION_C: dict = {
    "code": 1001,
    "controller": 0,
    "location": 0x10,
    "sequence": 0,
    "category": 5,
    "desc": 99,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _select_n(f: ActionLoopFilter, action: dict, n: int) -> None:
    """Call record_selection(action) n times."""
    for _ in range(n):
        f.record_selection(action)


def _make_filter_with_mock_env(
    state_changes: bool = False,
) -> tuple[ActionLoopFilter, MagicMock]:
    """Create an ActionLoopFilter backed by a MagicMock environment.

    Args:
        state_changes: If True, query_location returns a different card list
            on each call so the game-state fingerprint changes between snapshots.
    """
    env = MagicMock()

    if not state_changes:
        env._duel.query_location.return_value = []
    else:
        call_count = 0

        def _unique_per_call(player, loc):
            nonlocal call_count
            call_count += 1
            return [{"code": call_count, "sequence": 0, "position": 1, "status": 0}]

        env._duel.query_location.side_effect = _unique_per_call

    env._duel.game_state.lp = [8000, 8000]
    env._duel.game_state.phase = 0

    f = ActionLoopFilter(env)
    return f, env


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestActionKey:
    def test_extracts_all_fields(self):
        """action_key extracts the six identifying fields from an action dict."""
        action = {
            "code": 89631139,
            "controller": 0,
            "location": 0x04,
            "sequence": 2,
            "category": 1,
            "desc": 42,
        }
        assert ActionLoopFilter.action_key(action) == (89631139, 0, 0x04, 2, 1, 42)

    def test_defaults_missing_fields_to_zero(self):
        """Missing fields default to 0."""
        assert ActionLoopFilter.action_key({}) == (0, 0, 0, 0, 0, 0)

    def test_partial_action(self):
        """Only some fields present — rest default to 0."""
        action = {"code": 100, "location": 0x08}
        assert ActionLoopFilter.action_key(action) == (100, 0, 0x08, 0, 0, 0)


class TestRecordSelection:
    def test_interleaved_looping_actions_both_suppress(self):
        """Alternating between two looping actions suppresses both independently."""
        f, _ = _make_filter_with_mock_env()
        for _ in range(LOOP_DETECTION_THRESHOLD):
            f.record_selection(_ACTION_A)
            f.record_selection(_ACTION_B)
        assert f.is_looping(_ACTION_A)
        assert f.is_looping(_ACTION_B)

    def test_below_threshold_no_suppression(self):
        """Selecting the same key fewer than threshold times does not suppress."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD - 1)
        assert not f.is_looping(_ACTION_A)

    def test_interleaved_action_does_not_reset_count(self):
        """Selecting a different key does not reset another key's loop count."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD - 1)
        f.record_selection(_ACTION_B)
        # One more A selection reaches threshold (count was 2, now 3 = threshold)
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)

    def test_different_desc_is_different_action(self):
        """Keys that differ only in desc are tracked independently."""
        f, _ = _make_filter_with_mock_env()
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)
        assert not f.is_looping(_ACTION_C)


class TestLoopSuppression:
    def test_suppresses_on_unchanged_game_state(self):
        """threshold selections with no net change → suppressed."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)

    def test_no_suppression_on_net_change(self):
        """threshold selections with net change → not suppressed."""
        f, _ = _make_filter_with_mock_env(state_changes=True)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert not f.is_looping(_ACTION_A)

    def test_net_change_lifts_suppression(self):
        """Selecting the looping key when game state has changed lifts suppression."""
        queries_per_snapshot = 2 * len(_ZONES)

        env = MagicMock()
        call_count = 0

        def _stable_then_changes(player, loc):
            nonlocal call_count
            call_count += 1
            # Deferred fingerprinting: "before" at seen==threshold-1,
            # "after" at seen==threshold → 2 snapshots to suppress.
            # Third snapshot (the lift check) should differ.
            limit = 2 * queries_per_snapshot
            if call_count <= limit:
                return [{"code": 1, "sequence": 0, "position": 0, "status": 0}]
            return [{"code": 2, "sequence": 0, "position": 0, "status": 0}]

        env._duel.query_location.side_effect = _stable_then_changes
        env._duel.game_state.lp = [8000, 8000]
        env._duel.game_state.phase = 0

        f = ActionLoopFilter(env)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)

        # Game state changes — next selection of A lifts suppression
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

    def test_other_action_does_not_lift_suppression(self):
        """Selecting a different key does not affect a looping key."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)
        f.record_selection(_ACTION_B)
        assert f.is_looping(_ACTION_A)

    def test_looping_key_not_confused_with_other(self):
        """is_looping returns False for a different key when A is looping."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)
        assert not f.is_looping(_ACTION_B)
        assert not f.is_looping(_ACTION_C)


class TestNetChangeRetry:
    def test_new_baseline_captured_after_net_change(self):
        """After a net change, the new game-state snapshot becomes the baseline."""
        # Each fingerprint snapshot queries 2 players × len(_ZONES) zones.
        queries_per_snapshot = 2 * len(_ZONES)

        env = MagicMock()
        call_count = 0

        def _changes_then_stable(player, loc):
            nonlocal call_count
            call_count += 1
            # First snapshot: return card code=1
            # Second snapshot: return card code=2 (net change)
            # Third snapshot onward: return card code=3 (stable)
            if call_count <= queries_per_snapshot:
                return [{"code": 1, "sequence": 0, "position": 0, "status": 0}]
            elif call_count <= 2 * queries_per_snapshot:
                return [{"code": 2, "sequence": 0, "position": 0, "status": 0}]
            else:
                return [{"code": 3, "sequence": 0, "position": 0, "status": 0}]

        env._duel.query_location.side_effect = _changes_then_stable
        env._duel.game_state.lp = [8000, 8000]
        env._duel.game_state.phase = 0

        f = ActionLoopFilter(env)
        # threshold-1 calls → captures "before" (snapshot 1, all code=1)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD - 1)
        assert not f.is_looping(_ACTION_A)

        # threshold → captures "after" (snapshot 2, all code=2) → net change → not looping
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

        # threshold+1 → captures "after" (snapshot 3, all code=3) vs
        # stored "before" from previous (code=2) → net change → still not looping
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

        # threshold+2 → captures "after" (snapshot 4, all code=3) vs
        # stored "before" (code=3) → no net change → looping!
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)


class TestReset:
    def test_reset_clears_all_state(self):
        """reset() clears loop detection state and pending fingerprints."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)

        f.reset()

        assert not f.is_looping(_ACTION_A)
        assert not f._seen
        assert not f._pending_fp
        assert not f.has_looping_actions


class TestGameStateFingerprint:
    def test_no_duel_returns_empty_fingerprint(self):
        """_game_state_fingerprint() returns () when env._duel is None."""
        env = MagicMock()
        env._duel = None
        f = ActionLoopFilter(env)
        assert f._game_state_fingerprint() == ()

    def test_no_duel_detects_loop_on_repeated_selection(self):
        """With no duel, repeated selection still detects loop (empty == empty)."""
        env = MagicMock()
        env._duel = None
        f = ActionLoopFilter(env)
        _select_n(f, _ACTION_A, LOOP_DETECTION_THRESHOLD)
        assert f.is_looping(_ACTION_A)


# ─── Integration test (requires engine) ─────────────────────────────────────


def test_loop_filter_suppresses_disabled_quillbolt(lib, db_path, script_dirs, deck_path):
    """Disabled Quillbolt Hedgehog in GY is suppressed as a controlled loop.

    Minimal board: Tuner on field, disabled Quillbolt Hedgehog in GY.  The GY
    activation is legal but resolves with no net change to the game state.
    After threshold selections the filter detects the loop and removes the
    action from the next IDLE_CMD.

    Both agent and opponent actions are encoded in a GameRecording and
    replayed via cursor.
    """
    QUILLBOLT_HEDGEHOG = 23571046
    JUNK_SYNCHRON = 63977008

    DISABLE_QUILLBOLT_P0_LUA = b"""\
do
  local e=Effect.GlobalEffect()
  e:SetType(EFFECT_TYPE_FIELD)
  e:SetCode(EFFECT_DISABLE)
  e:SetTargetRange(LOCATION_GRAVE,0)
  e:SetTarget(function(e,c)
    return c:GetControler()==0
       and c:IsCode(23571046)
       and c:IsLocation(LOCATION_GRAVE)
  end)
  Duel.RegisterEffect(e,0)
end
"""

    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": JUNK_SYNCHRON, "pos": "face_up_attack", "seq": 0},
            ],
            "grave": [QUILLBOLT_HEDGEHOG],
        },
    }

    def _inject_gy_disable(state):
        base = generate_disable_lua(state) or ""
        return base + DISABLE_QUILLBOLT_P0_LUA.decode("utf-8")

    # Build interleaved recording: agent-only (opponent has no cards).
    # Each cycle: IDLE(activate Quillbolt Hedgehog) → CHAIN(decline) → CHAIN(decline).
    recording = GameRecording(setup={"agent_player": 0})
    for _ in range(LOOP_DETECTION_THRESHOLD):
        recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=1, num_actions=3)
        recording.append(msg_type=MSG_SELECT_CHAIN, player=0, action=0, num_actions=1)
        recording.append(msg_type=MSG_SELECT_CHAIN, player=0, action=0, num_actions=1)
    # Quillbolt Hedgehog loop is now detected; only Junk Synchron + end turn remain.
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=0, num_actions=2)

    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 42,
    }
    env = YuGiOhEnvironment(config)
    cursor = recording.cursor()
    env.set_opponent(ScriptedOpponent(cursor))
    try:
        with patch("yugioh_env.duel.generate_disable_lua", _inject_gy_disable):
            obs = env.reset(puzzle=puzzle, agent_player=0)

        for _ in range(3 * LOOP_DETECTION_THRESHOLD):
            entry = cursor.next_agent_entry(expected_msg_type=env._mapper.msg_type)
            obs = env.step(YuGiOhAction(action_index=entry["action"]))

        # Quillbolt Hedgehog loop detected; verify the next prompt
        # matches the recorded IDLE with 2 actions.
        num_actions = sum(obs.action_mask)
        entry = cursor.next_agent_entry(
            expected_msg_type=env._mapper.msg_type,
            expected_num_actions=num_actions,
        )
        assert entry["msg_type"] == MSG_SELECT_IDLECMD
        assert entry["num_actions"] == 2
    finally:
        env.close()


def test_loop_filter_suppresses_opponent_formula_synchron(lib, db_path, script_dirs, deck_path):
    """Disabled Formula Synchron's synchro effect is suppressed as a controlled loop.

    Covers the opponent branch of _apply_loop_filter.  Agent (player 0)
    has an empty board and ends turn.  Opponent (player 1) has a disabled
    Formula Synchron and Stardust Dragon on the field, with Shooting
    Star Dragon in the extra deck.  During the agent's main phase, the
    opponent repeatedly activates Formula Synchron's synchro summon
    effect via the chain window.  After threshold activations the
    filter detects the loop and suppresses it.

    Both agent and opponent actions are encoded in a GameRecording and
    replayed via cursor.
    """
    FORMULA_SYNCHRON = 50091196
    STARDUST_DRAGON = 44508094
    SHOOTING_STAR_DRAGON = 24696097

    puzzle = {
        "player1": {
            "monster_zone": [
                {"code": FORMULA_SYNCHRON, "pos": "face_up_attack", "seq": 0, "disabled": True},
                {"code": STARDUST_DRAGON, "pos": "face_up_attack", "seq": 1},
            ],
            "extra": [SHOOTING_STAR_DRAGON],
        },
    }

    # Build interleaved recording with both agent and opponent actions.
    # Agent has no cards — IDLE has 1 action (end turn).  After agent
    # ends turn, the opponent's chain window offers Formula Synchron
    # (2 options: activate or decline).  Each activation produces 2
    # agent chain declines.
    #
    # Sequence:
    #   agent IDLE(1, end turn)
    #   (opponent CHAIN(2, activate) → agent CHAIN(1, decline) × 2) × threshold
    #
    # After the last activation, the opponent's next chain window has
    # Formula Synchron suppressed (1 action, auto-played).  The agent
    # reaches its next IDLE without any opponent chain interruption.
    recording = GameRecording(setup={"agent_player": 0})
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=0, num_actions=1)
    for _ in range(LOOP_DETECTION_THRESHOLD):
        recording.append(
            msg_type=MSG_SELECT_CHAIN,
            player=1,
            action=0,
            num_actions=2,
        )
        recording.append(msg_type=MSG_SELECT_CHAIN, player=0, action=0, num_actions=1)
        recording.append(msg_type=MSG_SELECT_CHAIN, player=0, action=0, num_actions=1)
    # Agent reaches next IDLE — Formula Synchron loop is suppressed so no
    # opponent chain prompt intervenes.
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=0, num_actions=1)

    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
    }
    env = YuGiOhEnvironment(config)
    cursor = recording.cursor()
    env.set_opponent(ScriptedOpponent(cursor))
    try:
        obs = env.reset(puzzle=puzzle, agent_player=0, seed=42)
        assert not obs.done
        # Drive agent actions from the cursor
        agent_steps = 1 + 2 * LOOP_DETECTION_THRESHOLD
        for _ in range(agent_steps):
            entry = cursor.next_agent_entry(expected_msg_type=env._mapper.msg_type)
            obs = env.step(YuGiOhAction(action_index=entry["action"]))
        assert not obs.done
        assert env._loop_filter.has_looping_actions
        # Verify Formula Synchron loop is suppressed: the next prompt matches
        # the recorded IDLE — no opponent chain prompt intervened.
        num_actions = sum(obs.action_mask)
        entry = cursor.next_agent_entry(
            expected_msg_type=env._mapper.msg_type,
            expected_num_actions=num_actions,
        )
        assert entry["msg_type"] == MSG_SELECT_IDLECMD
    finally:
        env.close()
