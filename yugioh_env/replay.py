"""Deterministic game recording and playback for YuGiOhEnvironment.

Records both agent and opponent actions in an interleaved list so that
a game can be replayed deterministically given the same setup (seed,
decks, agent_player).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yugioh_env.models import YuGiOhAction, YuGiOhObservation
from yugioh_env.opponent import Inference, Opponent


class ReplayCursor:
    """Shared cursor over an interleaved action list for replay.

    All drift detection is consolidated in ``_verify``, called by both
    ``next_agent_entry`` and ``next_opponent_entry``.
    """

    def __init__(self, actions: list[dict[str, int]], agent_player: int) -> None:
        self._actions = actions
        self._agent_player = agent_player
        self._pos = 0

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._actions)

    def next_agent_entry(
        self,
        expected_msg_type: int | None = None,
        expected_num_actions: int | None = None,
    ) -> dict[str, int]:
        """Return next entry, verify it belongs to the agent.

        Optionally verify msg_type and num_actions match.
        Raises ``RuntimeError`` on drift.
        """
        entry = self._next()
        self._verify(entry, self._agent_player, expected_msg_type, expected_num_actions)
        return entry

    def next_opponent_entry(
        self,
        expected_msg_type: int | None = None,
        expected_num_actions: int | None = None,
    ) -> dict[str, int]:
        """Return next entry, verify it belongs to the opponent.

        Optionally verify msg_type and num_actions match.
        Raises ``RuntimeError`` on drift.
        """
        entry = self._next()
        opponent_player = 1 - self._agent_player
        self._verify(entry, opponent_player, expected_msg_type, expected_num_actions)
        return entry

    def _verify(
        self,
        entry: dict[str, int],
        expected_player: int,
        expected_msg_type: int | None,
        expected_num_actions: int | None,
    ) -> None:
        """Check player, msg_type, and num_actions."""
        step = self._pos - 1
        if entry["player"] != expected_player:
            raise RuntimeError(
                f"Replay drift at step {step}: "
                f"expected player {expected_player}, got player {entry['player']}"
            )
        if expected_msg_type is not None and entry["msg_type"] != expected_msg_type:
            raise RuntimeError(
                f"Replay drift at step {step}: "
                f"expected msg_type={expected_msg_type}, "
                f"got msg_type={entry['msg_type']}"
            )
        if expected_num_actions is not None and entry["num_actions"] != expected_num_actions:
            raise RuntimeError(
                f"Replay drift at step {step}: "
                f"expected num_actions={expected_num_actions}, "
                f"got num_actions={entry['num_actions']}"
            )

    def _next(self) -> dict[str, int]:
        if self.exhausted:
            raise RuntimeError(f"Replay action list exhausted at step {self._pos}")
        entry = self._actions[self._pos]
        self._pos += 1
        return entry


class GameRecording:
    """Data model for a recorded game.

    Stores the setup metadata (seed, decks, agent_player) and an ordered
    list of action entries ``{"msg_type": int, "player": int,
    "action": int, "num_actions": int}``.
    """

    def __init__(self, setup: dict[str, Any]) -> None:
        self.setup = setup
        self.actions: list[dict[str, int]] = []

    def append(self, *, msg_type: int, player: int, action: int, num_actions: int) -> None:
        """Append an action entry to the recording."""
        self.actions.append(
            {
                "msg_type": msg_type,
                "player": player,
                "action": action,
                "num_actions": num_actions,
            }
        )

    def cursor(self, agent_player: int | None = None) -> ReplayCursor:
        """Create a replay cursor. Defaults to setup["agent_player"]."""
        if agent_player is None:
            agent_player = self.setup["agent_player"]
        return ReplayCursor(self.actions, agent_player=agent_player)

    def save(self, path: str | Path) -> None:
        """Serialize the recording to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"setup": self.setup, "actions": self.actions}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GameRecording:
        """Deserialize a recording from JSON."""
        with open(path) as f:
            data = json.load(f)
        rec = cls(setup=data["setup"])
        rec.actions = data["actions"]
        return rec


class RecordingOpponent(Opponent):
    """Wraps any ``Opponent`` and records each action into a ``GameRecording``.

    ``seat_fn`` is a getter, not a fixed int: this wrapper is constructed
    before the env resolves ``agent_player`` for the episode, so at
    construction the attribute still holds the previous episode's seat (or the
    config default). Under ``agent_player="random"`` a snapshot taken there is
    wrong about half the time, and every recorded entry would carry the wrong
    seat -- which silently corrupts the drift detection replay depends on.
    """

    def __init__(
        self, inner: Opponent, recording: GameRecording, *, seat_fn: Callable[[], int]
    ) -> None:
        self._inner = inner
        self._recording = recording
        self._seat_fn = seat_fn

    @property
    def needs_board_state(self) -> bool:
        return self._inner.needs_board_state

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        action, inference = self._inner.select_action(obs)
        self._recording.append(
            msg_type=obs.msg_type,
            player=self._seat_fn(),
            action=action,
            num_actions=obs.num_actions,
        )
        return action, inference

    def reseed(self, seed: int) -> None:
        self._inner.reseed(seed)


class ScriptedOpponent(Opponent):
    """Plays actions from a ReplayCursor. Raises RuntimeError on drift."""

    @property
    def needs_board_state(self) -> bool:
        return False  # replays recorded indices, never reads the board

    def __init__(self, cursor: ReplayCursor) -> None:
        self._cursor = cursor

    def select_action(self, obs: YuGiOhObservation) -> tuple[int, Inference | None]:
        entry = self._cursor.next_opponent_entry(
            expected_msg_type=obs.msg_type,
            expected_num_actions=obs.num_actions,
        )
        return entry["action"], None

    def reseed(self, seed: int) -> None:
        pass  # No RNG to seed


class RecordingEnvironment:
    """Wraps a ``YuGiOhEnvironment`` and records agent actions.

    Agent actions are recorded in ``step()``, opponent actions are recorded
    via a ``RecordingOpponent`` wrapper installed on the inner environment.
    Both streams appear interleaved in a single ``GameRecording.actions`` list.
    """

    def __init__(self, env: Any, opponent: Opponent) -> None:
        self._env = env
        self._opponent = opponent
        self._recording: GameRecording | None = None
        self._done = False

    def reset(self, **kwargs: Any):
        """Start a new episode and begin recording."""
        setup = {
            k: kwargs[k]
            for k in ("seed", "deck0", "deck1", "agent_player", "puzzle")
            if k in kwargs
        }
        self._recording = GameRecording(setup)
        self._done = False

        recording_opponent = RecordingOpponent(
            self._opponent, self._recording, seat_fn=lambda: 1 - self._env._agent_player
        )
        self._env.set_opponent(recording_opponent)

        obs = self._env.reset(**kwargs)

        # Store the resolved agent_player (env resolves "random" internally)
        self._recording.setup["agent_player"] = self._env._agent_player

        return obs

    def step(self, action_index: int):
        """Execute an agent action, record it, and return the observation."""
        if self._done:
            return self._env.step(YuGiOhAction(action_index=action_index))

        # Record BEFORE stepping so the agent entry precedes any opponent
        # entries generated by _process_to_agent_choice inside env.step().
        msg_type = self._env._mapper.msg_type
        agent_player = self._env._agent_player
        num_actions = self._env.num_actions

        self._recording.append(
            msg_type=msg_type,
            player=agent_player,
            action=action_index,
            num_actions=num_actions,
        )

        obs = self._env.step(YuGiOhAction(action_index=action_index))
        if obs.done:
            self._done = True

        return obs

    def get_recording(self) -> GameRecording:
        """Return the current recording.

        Raises RuntimeError if no recording is active (reset not called).
        """
        if self._recording is None:
            raise RuntimeError("No active recording. Call reset() first.")
        return self._recording
