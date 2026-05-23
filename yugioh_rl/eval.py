"""Reusable evaluation primitives.

Powers two callers:

1. ``PPOTrainer._evaluate`` — periodic in-training eval. The trainer wraps
   its live ``YuGiOhNet`` in a ``NetworkOpponent`` and calls ``evaluate(...)``
   plus ``log_results_to_tensorboard(...)``.
2. The standalone eval CLI (added in Phase 3) — compares any two agents
   without a training loop.

The core loop drives an ``Opponent`` instance by reading ``env.current_msg`` /
``env.num_actions`` after each ``env.step()``; agents that need observations
also receive ``set_observation(obs)`` per step.

Agent reseeding: ``run_match`` reseeds the agent per episode, mirroring the
env-side reseed at ``yugioh_environment.py:reset()``. ``evaluate`` reseeds
from the same ``seed`` at the start of each opponent's match so cross-opponent
win rates compare against an identical seeded agent trajectory rather than a
drifting RNG stream.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from yugioh_env.opponent import (
    NetworkOpponent,
    Opponent,
    make_opponent,
    parse_opponent_spec,
)
from yugioh_rl.actor_learner import WorkerDiedError, WorkerTimeoutError
from yugioh_rl.env_wrapper import DeckDict, TrainingEnv

logger = logging.getLogger(__name__)


_DEFAULT_EVAL_WORKER_TIMEOUT_S = 1800.0   # 30 min — generous for model: opponents on CPU

# Pipe protocol — module-level constants so call sites can't typo a command.
_CMD_TASK = "task"
_CMD_SHUTDOWN = "shutdown"
_REPLY_PARTIAL = "partial"
_REPLY_ERROR = "error"


class _Worker(NamedTuple):
    """Pool driver's per-worker handle: parent-side pipe end + process."""
    conn: "mp.connection.Connection"
    proc: "mp.Process"


@dataclass
class EvalResult:
    """Win-rate breakdown for one (agent, opponent) match."""

    opponent_label: str
    episodes: int
    wins: int
    win_rate: float
    per_deck_wins: dict[int, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _EvalTask:
    """One unit of work for a parallel-eval worker: play episode ``episode_idx``
    of opponent ``opp_idx`` (1-indexed to match TrainingEnv's ``_episode_count``)."""

    opp_idx: int
    opp_spec: str
    episode_idx: int


class _PartialResult(NamedTuple):
    opp_idx: int
    episode_idx: int
    win: bool
    agent_deck_idx: int


class EvalWorkerError(RuntimeError):
    """A parallel-eval worker raised inside ``_eval_worker``.

    Args: ``(opp_spec, child_traceback_str)``.  Distinct from
    :class:`yugioh_rl.actor_learner.WorkerDiedError` (worker exited) and
    :class:`yugioh_rl.actor_learner.WorkerTimeoutError` (alive but silent).
    """

    def __str__(self) -> str:
        if len(self.args) == 2:
            return f"opponent {self.args[0]!r}: {self.args[1]}"
        return super().__str__()


def _build_tasks(opponent_specs: list[str], num_episodes: int) -> list[_EvalTask]:
    """Enumerate (opponent, episode_idx) tasks in opponent-major order.

    The nested-loop ordering naturally batches by opponent so a worker
    receives all of opponent A's tasks before any of opponent B's,
    minimizing env rebuilds per worker.
    """
    return [
        _EvalTask(opp_idx, spec, ep)
        for opp_idx, spec in enumerate(opponent_specs)
        for ep in range(1, num_episodes + 1)
    ]


def _aggregate_partials(
    partials: list[_PartialResult],
    opponent_specs: list[str],
) -> list[EvalResult]:
    """Group partials by ``opp_idx`` and build per-opponent ``EvalResult``s.

    Output order matches ``opponent_specs``.  Within each opponent, partials
    are sorted by ``episode_idx`` before assembling ``per_deck_wins`` so the
    list ordering is deterministic regardless of worker reply order — without
    that, two parallel runs at different worker counts could produce the same
    aggregate counts but differently-ordered per_deck lists, breaking the
    parity test.
    """
    by_opp: dict[int, list[_PartialResult]] = {}
    for p in partials:
        by_opp.setdefault(p.opp_idx, []).append(p)

    results: list[EvalResult] = []
    for opp_idx, spec in enumerate(opponent_specs):
        opp_parts = sorted(by_opp.get(opp_idx, []), key=lambda p: p.episode_idx)
        wins = sum(1 for p in opp_parts if p.win)
        episodes = len(opp_parts)
        per_deck: dict[int, list[float]] = {}
        for p in opp_parts:
            per_deck.setdefault(p.agent_deck_idx, []).append(1.0 if p.win else 0.0)
        results.append(EvalResult(
            opponent_label=opponent_label_from_spec(spec),
            episodes=episodes,
            wins=wins,
            win_rate=(wins / episodes) if episodes > 0 else 0.0,
            per_deck_wins=per_deck,
        ))
    return results


def opponent_label_from_spec(spec: str) -> str:
    """Human-readable label used in TensorBoard scalar keys and console logs.

    Bare specs (``"greedy"`` / ``"random"``) pass through as-is.
    ``"model:/a/b/c.pt"`` becomes ``"model_b_c"`` (parent dir + stem).
    ``"model:c.pt"`` (no parent) becomes ``"model_c"``.
    """
    opp_type, checkpoint = parse_opponent_spec(spec)
    if opp_type == "model":
        p = Path(checkpoint)
        parent = p.parent.name
        return f"model_{parent}_{p.stem}" if parent else f"model_{p.stem}"
    return opp_type


def make_eval_agent(
    spec: str,
    *,
    seed: int = 0,
    device: str = "cpu",
    network=None,
) -> Opponent:
    """Build an ``Opponent`` instance for the agent-side of an eval.

    When ``network`` is provided, returns a ``NetworkOpponent`` and ignores
    ``spec`` — this is the in-training path that avoids a checkpoint reload.
    Otherwise delegates to ``yugioh_env.opponent.make_opponent`` so the
    spec-string contract (parsing + error messages) stays in one place.

    The returned agent is reseeded per-episode by ``run_match``; ``seed``
    here only sets the initial state.
    """
    if network is not None:
        return NetworkOpponent(network, device=device)
    return make_opponent(spec, seed=seed, device=device)


def _play_one_episode(
    agent: Opponent,
    env: TrainingEnv,
    *,
    base_seed: int,
    episode_idx: int,
) -> tuple[bool, int]:
    """Play episode index ``episode_idx`` (1-indexed to match _episode_count).

    Reseeds the agent from ``base_seed + episode_idx`` and resets the env
    via ``reset(episode_idx=episode_idx)`` so the body is usable both by
    sequential ``run_match`` (idx=1..N in order) and by parallel-eval
    workers dispatching episodes by absolute index.

    Returns ``(win, agent_deck_idx)`` from the terminal info dict.
    """
    agent.reseed(base_seed + episode_idx)
    obs = env.reset(episode_idx=episode_idx)
    done = False
    info: dict[str, Any] = {}
    while not done:
        if agent.needs_observation:
            agent.set_observation(obs)
        action = agent.select_action(env.current_msg, env.num_actions)
        obs, _reward, done, info = env.step(action)
    win = info.get("terminal_reward", 0) > 0
    return win, int(info.get("agent_deck_idx", 0))


def run_match(
    agent: Opponent,
    env: TrainingEnv,
    num_episodes: int,
    *,
    base_seed: int,
) -> tuple[int, dict[int, list[float]]]:
    """Run ``num_episodes`` against the env's pre-configured opponent.

    Returns ``(total_wins, per_deck)`` where ``per_deck`` maps
    ``agent_deck_idx`` → list of 1.0/0.0 win records.
    """
    total_wins = 0
    per_deck: dict[int, list[float]] = {}
    # num_episodes == 0 is a valid "skip" signal; don't pay a duel-init cost
    # just to play zero episodes (matches the pre-refactor for-loop semantics).
    if num_episodes <= 0:
        return total_wins, per_deck
    for i in range(num_episodes):
        win, deck_idx = _play_one_episode(
            agent, env, base_seed=base_seed, episode_idx=i + 1,
        )
        if win:
            total_wins += 1
        per_deck.setdefault(deck_idx, []).append(1.0 if win else 0.0)
    return total_wins, per_deck


def _eval_worker(
    remote,
    *,
    agent_spec: str,
    agent_device: str,
    deck_pool: list[DeckDict],
    seed: int,
    agent_player: str,
    opponent_device: str | None,
) -> None:
    """Parallel-eval worker process: holds one ``TrainingEnv`` cached by
    opponent_spec, plays one episode per ``("task", _EvalTask)`` message,
    replies ``("partial", _PartialResult)`` or ``("error", traceback_str)``.

    Mirrors :func:`yugioh_rl.actor_learner._actor_learner_worker`'s pipe
    protocol and shutdown handshake.
    """
    # Deferred imports keep the spawn-context module load minimal.
    from yugioh_rl.eval import _play_one_episode, make_eval_agent
    from yugioh_rl.env_wrapper import TrainingEnv as _TrainingEnv

    env: _TrainingEnv | None = None
    current_spec: str | None = None
    try:
        agent = make_eval_agent(agent_spec, seed=seed, device=agent_device)
        while True:
            cmd, payload = remote.recv()
            if cmd == _CMD_SHUTDOWN:
                break
            assert cmd == _CMD_TASK, f"unknown cmd {cmd!r}"
            task: _EvalTask = payload
            try:
                if task.opp_spec != current_spec:
                    if env is not None:
                        env.close()
                    env = _TrainingEnv(**_make_eval_env_kwargs(
                        deck_pool, task.opp_spec,
                        seed=seed, agent_player=agent_player,
                        opponent_device=opponent_device,
                    ))
                    current_spec = task.opp_spec

                win, agent_deck_idx = _play_one_episode(
                    agent, env, base_seed=seed, episode_idx=task.episode_idx,
                )
                remote.send((
                    _REPLY_PARTIAL,
                    _PartialResult(task.opp_idx, task.episode_idx, win, agent_deck_idx),
                ))
            except Exception:
                remote.send((_REPLY_ERROR, traceback.format_exc()))
                return
    except Exception:
        remote.send((_REPLY_ERROR, traceback.format_exc()))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_eval_pool(
    *,
    agent_spec: str,
    agent_device: str,
    opponent_specs: list[str],
    deck_pool: list[DeckDict],
    num_episodes: int,
    seed: int,
    agent_player: str,
    opponent_device: str | None,
    num_workers: int,
    worker_timeout_s: float,
    worker_fn=None,
) -> list[EvalResult]:
    """Fan ``opponent_specs × num_episodes`` out across worker processes.

    ``worker_fn`` defaults to :func:`_eval_worker`; tests inject fake worker
    targets to exercise the error taxonomy in isolation.  Must be a
    module-level callable so the spawn-context can pickle it.
    """
    tasks = _build_tasks(opponent_specs, num_episodes)
    if not tasks:
        return _aggregate_partials([], opponent_specs)
    K = min(num_workers, len(tasks))
    ctx = mp.get_context("spawn")
    target = worker_fn if worker_fn is not None else _eval_worker

    workers: list[_Worker] = []
    for _ in range(K):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=target,
            kwargs={
                "remote": child_conn,
                "agent_spec": agent_spec,
                "agent_device": agent_device,
                "deck_pool": deck_pool,
                "seed": seed,
                "agent_player": agent_player,
                "opponent_device": opponent_device,
            },
            daemon=True,
        )
        proc.start()
        child_conn.close()
        workers.append(_Worker(parent_conn, proc))

    def _send_task(w_idx: int, task: _EvalTask) -> None:
        """Send a task, converting pipe death into the same WorkerDiedError
        taxonomy used on the recv path.  Without this, a worker that exits
        between accepting two tasks surfaces as a raw ``BrokenPipeError``.
        """
        worker = workers[w_idx]
        try:
            worker.conn.send((_CMD_TASK, task))
        except (BrokenPipeError, EOFError, ConnectionResetError):
            worker.proc.join(timeout=0.5)
            raise WorkerDiedError(
                f"eval worker pid={worker.proc.pid} died "
                f"(exitcode={worker.proc.exitcode}) before accepting task"
            ) from None

    partials: list[_PartialResult] = []
    try:
        task_iter = iter(tasks)
        outstanding: dict[int, _EvalTask] = {}

        for w_idx in range(len(workers)):
            try:
                t = next(task_iter)
            except StopIteration:
                break
            _send_task(w_idx, t)
            outstanding[w_idx] = t

        while outstanding:
            ready_conns = mp.connection.wait(
                [workers[i].conn for i in outstanding],
                timeout=worker_timeout_s,
            )
            if not ready_conns:
                # Distinguish died-but-pipe-not-yet-ready from alive-but-silent,
                # mirroring actor_learner.py:291.
                for w_idx in list(outstanding):
                    proc = workers[w_idx].proc
                    if not proc.is_alive():
                        raise WorkerDiedError(
                            f"eval worker pid={proc.pid} died "
                            f"(exitcode={proc.exitcode}) before reply"
                        )
                raise WorkerTimeoutError(
                    f"eval workers alive but silent for {worker_timeout_s:.0f}s"
                )
            for conn in ready_conns:
                w_idx = next(i for i, w in enumerate(workers) if w.conn is conn)
                proc = workers[w_idx].proc
                try:
                    cmd, payload = conn.recv()
                except (EOFError, ConnectionResetError):
                    raise WorkerDiedError(
                        f"eval worker pid={proc.pid} died "
                        f"(exitcode={proc.exitcode}) mid-task"
                    ) from None
                if cmd == _REPLY_ERROR:
                    raise EvalWorkerError(outstanding[w_idx].opp_spec, payload)
                assert cmd == _REPLY_PARTIAL, f"unexpected cmd {cmd!r}"
                partials.append(payload)
                try:
                    t = next(task_iter)
                except StopIteration:
                    del outstanding[w_idx]
                else:
                    _send_task(w_idx, t)
                    outstanding[w_idx] = t
    finally:
        for worker in workers:
            try:
                worker.conn.send((_CMD_SHUTDOWN, None))
            except (BrokenPipeError, EOFError):
                pass
        for worker in workers:
            worker.proc.join(timeout=5)
            if worker.proc.is_alive():
                worker.proc.terminate()

    return _aggregate_partials(partials, opponent_specs)


def _make_eval_env_kwargs(
    deck_pool: list[DeckDict],
    opponent_spec: str,
    *,
    seed: int,
    agent_player: str,
    opponent_device: str | None,
) -> dict[str, Any]:
    """Build the kwargs dict for an eval-side ``TrainingEnv``.

    ``opponent_device`` is omitted when None so the ``YUGIOH_OPPONENT_DEVICE``
    env-var fallback inside ``YuGiOhEnvironment`` keeps working.
    """
    kwargs: dict[str, Any] = {
        "deck_pool": deck_pool,
        "opponent": opponent_spec,
        "reward_shaping": False,
        "seed": seed,
        "agent_player": agent_player,
    }
    if opponent_device is not None:
        kwargs["opponent_device"] = opponent_device
    return kwargs


def _run_sequential_match_set(
    agent: Opponent,
    deck_pool: list[DeckDict],
    opponent_specs: list[str],
    *,
    num_episodes: int,
    seed: int,
    agent_player: str,
    opponent_device: str | None,
) -> list[EvalResult]:
    """Sequential per-opponent loop shared by both public entry points."""
    results: list[EvalResult] = []
    for spec in opponent_specs:
        env = TrainingEnv(**_make_eval_env_kwargs(
            deck_pool, spec,
            seed=seed, agent_player=agent_player,
            opponent_device=opponent_device,
        ))
        try:
            wins, per_deck = run_match(agent, env, num_episodes, base_seed=seed)
        finally:
            env.close()
        results.append(
            EvalResult(
                opponent_label=opponent_label_from_spec(spec),
                episodes=num_episodes,
                wins=wins,
                win_rate=wins / max(num_episodes, 1),
                per_deck_wins=per_deck,
            )
        )
    return results


def evaluate(
    agent_spec: str,
    deck_pool: list[DeckDict],
    opponent_specs: list[str],
    *,
    num_episodes: int,
    seed: int,
    agent_player: str = "random",
    opponent_device: str | None = None,
    workers: int = 1,
    agent_device: str = "cpu",
    worker_timeout_s: float = _DEFAULT_EVAL_WORKER_TIMEOUT_S,
) -> list[EvalResult]:
    """Run an agent (specified by string) against each opponent spec.

    ``agent_spec`` is any string accepted by :func:`make_eval_agent`:
    ``"random"`` / ``"greedy"`` / ``"model:path/to/ckpt.pt"``.  This is the
    form CLI and leaderboard callers should use — the spec is portable
    across both the sequential path (built in-process) and the parallel
    path (workers re-instantiate locally).

    For the in-training ``NetworkOpponent`` case (no string spec exists
    because the agent wraps a live ``nn.Module``), use
    :func:`evaluate_with_agent` instead.

    Sequential (``workers=1``): a fresh ``TrainingEnv`` per spec with the
    same ``seed`` / ``agent_player``; ``opponent_device`` is forwarded only
    when non-None to preserve the ``YUGIOH_OPPONENT_DEVICE`` fallback.

    Parallel (``workers>=2``): episode-level results are aggregated
    deterministically — runs at different ``workers`` counts produce
    byte-equal ``EvalResult``s.
    """
    if workers <= 1:
        agent = make_eval_agent(agent_spec, seed=seed, device=agent_device)
        return _run_sequential_match_set(
            agent, deck_pool, opponent_specs,
            num_episodes=num_episodes, seed=seed,
            agent_player=agent_player, opponent_device=opponent_device,
        )
    return _run_eval_pool(
        agent_spec=agent_spec,
        agent_device=agent_device,
        opponent_specs=opponent_specs,
        deck_pool=deck_pool,
        num_episodes=num_episodes,
        seed=seed,
        agent_player=agent_player,
        opponent_device=opponent_device,
        num_workers=workers,
        worker_timeout_s=worker_timeout_s,
    )


def evaluate_with_agent(
    agent: Opponent,
    deck_pool: list[DeckDict],
    opponent_specs: list[str],
    *,
    num_episodes: int,
    seed: int,
    agent_player: str = "random",
    opponent_device: str | None = None,
) -> list[EvalResult]:
    """Sequential eval with a pre-built :class:`Opponent` instance.

    Used by the in-training ``PPOTrainer._evaluate`` path that wraps a
    live ``nn.Module`` in :class:`NetworkOpponent` — no string spec exists
    for that, and the live module can't cross a spawn boundary, so this
    path is sequential-only.

    For string-spec callers (CLI, leaderboard), use :func:`evaluate`,
    which also supports parallel workers.
    """
    return _run_sequential_match_set(
        agent, deck_pool, opponent_specs,
        num_episodes=num_episodes, seed=seed,
        agent_player=agent_player, opponent_device=opponent_device,
    )


def log_results_to_tensorboard(
    writer,
    results: list[EvalResult],
    deck_paths: list[str],
    global_step: int,
) -> None:
    """Write eval/win_rate_vs_{label} and per-deck scalars.

    Key format must stay byte-identical to the pre-refactor output so
    existing TensorBoard runs continue without a metric split.
    """
    deck_stems = [Path(p).stem for p in deck_paths]
    for r in results:
        writer.add_scalar(
            f"eval/win_rate_vs_{r.opponent_label}", r.win_rate, global_step
        )
        for deck_idx, deck_results in r.per_deck_wins.items():
            deck_wr = sum(deck_results) / len(deck_results) if deck_results else 0.0
            writer.add_scalar(
                f"eval/win_rate_vs_{r.opponent_label}_deck_{deck_stems[deck_idx]}",
                deck_wr,
                global_step,
            )
