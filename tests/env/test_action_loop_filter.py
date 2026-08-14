"""Unit and integration tests for ActionLoopFilter."""

from __future__ import annotations

from unittest.mock import MagicMock

from yugioh_core.constants import (
    LOCATION_GRAVE,
    LOCATION_MZONE,
    LOCATION_SZONE,
    MSG_SELECT_CARD,
    MSG_SELECT_CHAIN,
    MSG_SELECT_IDLECMD,
    STATUS_DISABLED,
)
from yugioh_env.action_loop_filter import _ZONES, SAMPLING_START, ActionLoopFilter
from yugioh_env.models import YuGiOhAction
from yugioh_env.replay import GameRecording, ScriptedOpponent
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

# `_N_SUPPRESS` is the *selection* index at which a period-1 (unchanging-state)
# loop is suppressed: its state is first sampled (the baseline) at selection
# ``sampling_start``, and the next selection recurs → suppressed.  (A period-P
# loop needs P-1 more selections first to sample its other distinct states;
# these tests use P=1.)
_N_SUPPRESS = SAMPLING_START + 1

# ─── Generic test actions ────────────────────────────────────────────────────

_ACTION_A: dict = {
    "code": 1001,
    "controller": 0,
    "location": LOCATION_GRAVE,
    "sequence": 0,
    "category": 5,
    "desc": 0,
}
_ACTION_B: dict = {
    "code": 1002,
    "controller": 0,
    "location": LOCATION_MZONE,
    "sequence": 1,
    "category": 1,
    "desc": 0,
}
_ACTION_C: dict = {
    "code": 1001,
    "controller": 0,
    "location": LOCATION_GRAVE,
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
            "location": LOCATION_MZONE,
            "sequence": 2,
            "category": 1,
            "desc": 42,
        }
        assert ActionLoopFilter.action_key(action) == (89631139, 0, LOCATION_MZONE, 2, 1, 42)

    def test_defaults_missing_fields_to_zero(self):
        """Missing fields default to 0."""
        assert ActionLoopFilter.action_key({}) == (0, 0, 0, 0, 0, 0)

    def test_partial_action(self):
        """Only some fields present — rest default to 0."""
        action = {"code": 100, "location": LOCATION_SZONE}
        assert ActionLoopFilter.action_key(action) == (100, 0, 0x08, 0, 0, 0)


class TestRecordSelection:
    def test_interleaved_looping_actions_both_suppress(self):
        """Alternating between two looping actions suppresses both independently."""
        f, _ = _make_filter_with_mock_env()
        for _ in range(_N_SUPPRESS):
            f.record_selection(_ACTION_A)
            f.record_selection(_ACTION_B)
        assert f.is_looping(_ACTION_A)
        assert f.is_looping(_ACTION_B)

    def test_below_suppression_count_no_suppression(self):
        """Too few selections to reach a recurrence does not suppress."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS - 1)
        assert not f.is_looping(_ACTION_A)

    def test_interleaved_action_does_not_reset_count(self):
        """Selecting a different key does not reset another key's loop count."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS - 1)
        f.record_selection(_ACTION_B)
        # One more A selection reaches the suppression count (its state recurs).
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)

    def test_different_desc_is_different_action(self):
        """Keys that differ only in desc are tracked independently."""
        f, _ = _make_filter_with_mock_env()
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)
        assert not f.is_looping(_ACTION_C)


class TestLoopSuppression:
    def test_suppresses_on_unchanged_game_state(self):
        """An unchanging state recurs → suppressed once the recurrence count is hit."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)

    def test_no_suppression_on_net_change(self):
        """A new state every selection never recurs → not suppressed."""
        f, _ = _make_filter_with_mock_env(state_changes=True)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert not f.is_looping(_ACTION_A)

    def test_net_change_lifts_suppression(self):
        """Selecting the looping key when game state has changed lifts suppression."""
        queries_per_snapshot = 2 * len(_ZONES)

        env = MagicMock()
        call_count = 0

        def _stable_then_changes(player, loc):
            nonlocal call_count
            call_count += 1
            # Selection 1 is free; selections 2 and 3 see the same state
            # (baseline, then a recurrence) → suppressed. Selection 4
            # (call_count past the limit) returns a new state → lifts it.
            limit = 2 * queries_per_snapshot
            if call_count <= limit:
                return [{"code": 1, "sequence": 0, "position": 0, "status": 0}]
            return [{"code": 2, "sequence": 0, "position": 0, "status": 0}]

        env._duel.query_location.side_effect = _stable_then_changes
        env._duel.game_state.lp = [8000, 8000]
        env._duel.game_state.phase = 0

        f = ActionLoopFilter(env)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)

        # Game state changes — next selection of A lifts suppression
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

    def test_other_action_does_not_lift_suppression(self):
        """Selecting a different key does not affect a looping key."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)
        f.record_selection(_ACTION_B)
        assert f.is_looping(_ACTION_A)

    def test_looping_key_not_confused_with_other(self):
        """is_looping returns False for a different key when A is looping."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)
        assert not f.is_looping(_ACTION_B)
        assert not f.is_looping(_ACTION_C)


class TestNetChangeThenRecurrence:
    def test_suppresses_only_once_a_state_repeats(self):
        """A run of new states never suppresses; the first recurrence does."""
        # Each fingerprint snapshot queries 2 players × len(_ZONES) zones.
        queries_per_snapshot = 2 * len(_ZONES)

        env = MagicMock()
        call_count = 0

        def _changes_then_stable(player, loc):
            nonlocal call_count
            call_count += 1
            # Snapshot 1: code=1; snapshot 2: code=2; snapshot 3 onward: code=3.
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
        # Selection 1 is free; selection 2 samples code=1 (a new state).
        _select_n(f, _ACTION_A, _N_SUPPRESS - 1)
        assert not f.is_looping(_ACTION_A)

        # Selection 3 → code=2 (new state) → not looping.
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

        # Selection 4 → code=3 (new state) → not looping.
        f.record_selection(_ACTION_A)
        assert not f.is_looping(_ACTION_A)

        # Selection 5 → code=3 again → first recurrence → looping.
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)


def _make_filter_with_state_cycle(
    states: list[dict],
) -> tuple[ActionLoopFilter, MagicMock]:
    """ActionLoopFilter whose fingerprint cycles through *states*.

    One entry of *states* is used per fingerprint snapshot (a snapshot spans
    ``2 * len(_ZONES)`` query_location calls), cycling round-robin.  Each entry
    is the card dict returned by query_location for that snapshot.
    """
    env = MagicMock()
    qps = 2 * len(_ZONES)
    calls = 0

    def _q(player, loc):
        nonlocal calls
        snap = calls // qps
        calls += 1
        return [states[snap % len(states)]]

    env._duel.query_location.side_effect = _q
    env._duel.game_state.lp = [8000, 8000]
    env._duel.game_state.phase = 0
    return ActionLoopFilter(env), env


class TestRecurrenceDetection:
    """Recurrence detection catches period-N loops, not just period-1."""

    def test_period_2_loop_suppressed(self):
        """A state that alternates A,B,A,B is caught when A recurs."""
        f, _ = _make_filter_with_state_cycle(
            [
                {"code": 1, "sequence": 0, "position": 0, "status": 0},
                {"code": 2, "sequence": 0, "position": 0, "status": 0},
            ]
        )
        # First selections only sample the two distinct states (no recurrence yet).
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert not f.is_looping(_ACTION_A)
        # The next selection returns to the first state → recurrence → suppressed.
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)

    def test_period_3_loop_suppressed(self):
        """A state cycling A,B,C is caught when A recurs (period-N generality)."""
        f, _ = _make_filter_with_state_cycle(
            [
                {"code": 1, "sequence": 0, "position": 0, "status": 0},
                {"code": 2, "sequence": 0, "position": 0, "status": 0},
                {"code": 3, "sequence": 0, "position": 0, "status": 0},
            ]
        )
        _select_n(f, _ACTION_A, _N_SUPPRESS + 1)
        assert not f.is_looping(_ACTION_A)
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)

    def test_oscillating_disabled_bit_is_caught(self):
        """The real bug shape: only a card's STATUS_DISABLED bit flips each cycle.

        The old compare-to-previous filter never matched (consecutive states
        always differed); recurrence detection catches it when the bit returns
        to its earlier value.
        """
        f, _ = _make_filter_with_state_cycle(
            [
                {"code": 999, "sequence": 0, "position": 0, "status": 0x00},
                {"code": 999, "sequence": 0, "position": 0, "status": STATUS_DISABLED},
            ]
        )
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert not f.is_looping(_ACTION_A)
        f.record_selection(_ACTION_A)
        assert f.is_looping(_ACTION_A)


class TestReset:
    def test_reset_clears_all_state(self):
        """reset() clears loop detection state and pending fingerprints."""
        f, _ = _make_filter_with_mock_env(state_changes=False)
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)

        f.reset()

        assert not f.is_looping(_ACTION_A)
        assert not f._seen
        assert not f._fp_history
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
        _select_n(f, _ACTION_A, _N_SUPPRESS)
        assert f.is_looping(_ACTION_A)


# ─── Integration test (requires engine) ─────────────────────────────────────


def test_loop_filter_suppresses_disabled_quillbolt(lib, db_path, script_dirs, deck_path):
    """Disabled Quillbolt Hedgehog in GY is suppressed as a controlled loop.

    Minimal board: Tuner on field, disabled Quillbolt Hedgehog in GY.  The GY
    activation is legal but resolves with no net change to the game state.
    Once the state recurs the filter detects the loop and removes the action
    from the next IDLE_CMD.

    Both agent and opponent actions are encoded in a GameRecording and
    replayed via cursor.
    """
    QUILLBOLT_HEDGEHOG = 23571046
    JUNK_SYNCHRON = 63977008

    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": JUNK_SYNCHRON, "pos": "face_up_attack", "seq": 0},
            ],
            "grave": [{"code": QUILLBOLT_HEDGEHOG, "disabled": True}],
        },
    }

    # Build interleaved recording: agent-only (opponent has no cards).
    # Each cycle: IDLE(activate Quillbolt Hedgehog) → CHAIN(decline) → CHAIN(decline).
    recording = GameRecording(setup={"agent_player": 0})
    for _ in range(_N_SUPPRESS):
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
        obs = env.reset(puzzle=puzzle, agent_player=0)

        for _ in range(3 * _N_SUPPRESS):
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


def test_loop_filter_suppresses_period2_self_negate(lib, db_path, script_dirs, deck_path):
    """A period-2 (oscillating-STATUS_DISABLED) no-op loop is suppressed.

    Minimal engine repro of the non-terminating chain loop.  Alector, Sovereign
    of Birds targets *itself* → registers a self-owned EFFECT_DISABLE whose
    derived STATUS_DISABLED bit oscillates on every adjust_instant.  Honest is
    placed already-disabled; its unlimited, cost-free "return to hand" ignition
    resolves negated (handler disabled) → a repeatable no-op.  Spamming Honest
    leaves the board identical except Alector's flickering bit — a period-2
    loop the old compare-to-previous filter missed.  Recurrence detection now
    suppresses it on the loop's first recurrence.

    Agent (player 0) actions are replayed from a GameRecording via a cursor;
    the opponent (player 1) has no cards.  Sequence per collapse-forced play:
    IDLE(activate Alector, idx 2/5) → SELECT_CARD(target Alector itself, idx
    0/2) → IDLE(activate Honest, idx 2/4) × 4 (the period-2 loop's first
    recurrence lands on the 4th activation).  After the loop is suppressed,
    the next IDLE drops Honest (4 → 3 actions).
    """
    ALECTOR = 17573739
    HONEST = 37742478

    puzzle = {
        "player0": {
            "monster_zone": [
                {"code": ALECTOR, "pos": "face_up_attack", "seq": 0},
                {"code": HONEST, "pos": "face_up_attack", "seq": 1, "disabled": True},
            ],
        },
    }

    # Number of Honest activations until recurrence catches the period-2 loop.
    honest_reps = _N_SUPPRESS + 1

    recording = GameRecording(setup={"agent_player": 0})
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=2, num_actions=5)
    recording.append(msg_type=MSG_SELECT_CARD, player=0, action=0, num_actions=2)
    for _ in range(honest_reps):
        recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=2, num_actions=4)
    # Honest loop now suppressed — the next IDLE offers 3 actions (Honest gone).
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=0, num_actions=3)

    config = {
        "db_path": str(db_path),
        "script_dirs": [str(d) for d in script_dirs],
        "deck_path": str(deck_path),
        "opponent": "random",
        "opponent_seed": 1,
        "collapse_forced": True,
    }
    env = YuGiOhEnvironment(config)
    cursor = recording.cursor()
    env.set_opponent(ScriptedOpponent(cursor))
    try:
        obs = env.reset(puzzle=puzzle, agent_player=0)

        # Drive the recorded agent actions: arm (2) + Honest spam.
        for _ in range(2 + honest_reps):
            entry = cursor.next_agent_entry(expected_msg_type=env._mapper.msg_type)
            obs = env.step(YuGiOhAction(action_index=entry["action"]))

        # Honest's period-2 loop is detected and suppressed.
        assert env._loop_filter.has_looping_actions
        num_actions = sum(obs.action_mask)
        entry = cursor.next_agent_entry(
            expected_msg_type=env._mapper.msg_type,
            expected_num_actions=num_actions,
        )
        assert entry["msg_type"] == MSG_SELECT_IDLECMD
        assert entry["num_actions"] == 3
    finally:
        env.close()


def test_loop_filter_suppresses_opponent_formula_synchron(lib, db_path, script_dirs, deck_path):
    """Disabled Formula Synchron's synchro effect is suppressed as a controlled loop.

    Covers the opponent branch of _apply_loop_filter.  Agent (player 0)
    has an empty board and ends turn.  Opponent (player 1) has a disabled
    Formula Synchron and Stardust Dragon on the field, with Shooting
    Star Dragon in the extra deck.  During the agent's main phase, the
    opponent repeatedly activates Formula Synchron's synchro summon
    effect via the chain window.  Once the state recurs the filter
    detects the loop and suppresses it.

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
    #   (opponent CHAIN(2, activate) → agent CHAIN(1, decline) × 2) × _N_SUPPRESS
    #
    # After the last activation, the opponent's next chain window has
    # Formula Synchron suppressed (1 action, auto-played).  The agent
    # reaches its next IDLE without any opponent chain interruption.
    recording = GameRecording(setup={"agent_player": 0})
    recording.append(msg_type=MSG_SELECT_IDLECMD, player=0, action=0, num_actions=1)
    for _ in range(_N_SUPPRESS):
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
        agent_steps = 1 + 2 * _N_SUPPRESS
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
