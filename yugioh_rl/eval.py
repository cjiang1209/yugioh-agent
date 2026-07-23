"""Reusable evaluation primitives.

Powers two callers:

1. ``PPOTrainer._evaluate`` — periodic in-training eval. The trainer wraps
   its live ``YuGiOhNet`` in a ``NetworkOpponent``, calls ``evaluate(...)``,
   then routes the results through its logging sinks.
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
import statistics
import traceback
from contextlib import suppress
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
from yugioh_rl.env_wrapper import DeckDict, EvalEnv

logger = logging.getLogger(__name__)


_DEFAULT_EVAL_WORKER_TIMEOUT_S = 1800.0  # 30 min — generous for model: opponents on CPU

# Pipe protocol — module-level constants so call sites can't typo a command.
_CMD_TASK = "task"
_CMD_SHUTDOWN = "shutdown"
_REPLY_PARTIAL = "partial"
_REPLY_ERROR = "error"


class _Worker(NamedTuple):
    """Pool driver's per-worker handle: parent-side pipe end + process."""

    conn: mp.connection.Connection
    proc: mp.Process


@dataclass
class EvalResult:
    """Win-rate breakdown for one (agent, opponent) match."""

    opponent_label: str
    episodes: int
    wins: int
    per_deck_wins: dict[int, list[float]] = field(default_factory=dict)
    steps_mean: float = 0.0
    steps_std: float = 0.0
    steps_median: float = 0.0
    steps_max: int = 0
    turns_mean: float = 0.0
    turns_std: float = 0.0
    turns_median: float = 0.0
    turns_max: int = 0
    wins_first: int = 0
    episodes_first: int = 0
    wins_second: int = 0
    episodes_second: int = 0
    timeouts: int = 0

    @property
    def win_rate(self) -> float:
        """Fraction of episodes won — derived from ``wins``/``episodes``."""
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def play_first_rate(self) -> float:
        """Fraction of episodes the agent went first — derived; every episode is
        either first or second, so ``episodes_first + episodes_second == episodes``."""
        return self.episodes_first / self.episodes if self.episodes else 0.0


@dataclass(frozen=True)
class _EvalTask:
    """One unit of work for a parallel-eval worker: play episode ``episode_idx``
    of opponent ``opp_idx`` (1-indexed to match TrainingEnv's ``_episode_count``)."""

    opp_idx: int
    opp_spec: str
    episode_idx: int


class _EpisodeRecord(NamedTuple):
    """One completed episode's outcome, independent of any (opponent, worker)
    bookkeeping — the shared input type to :func:`_aggregate_one`."""

    episode_idx: int
    win: bool
    agent_deck_idx: int
    steps: int
    turns: int
    went_first: bool
    timeout: bool = False


class _PartialResult(NamedTuple):
    opp_idx: int
    episode_idx: int
    win: bool
    agent_deck_idx: int
    steps: int
    turns: int
    went_first: bool
    timeout: bool = False


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


def _aggregate_one(records, opponent_label: str) -> EvalResult:
    """Build an EvalResult from per-episode records (pre-sorted by episode_idx).

    Shared aggregator for both the sequential (``_run_sequential_match_set``)
    and parallel (``_aggregate_partials``) paths — the single place win/steps/
    turns/order-split math lives, so the two paths can't drift apart.
    """
    n = len(records)
    wins = sum(1 for r in records if r.win)
    per_deck: dict[int, list[float]] = {}
    for r in records:
        per_deck.setdefault(r.agent_deck_idx, []).append(1.0 if r.win else 0.0)

    def _stats(vals):
        if not vals:
            return 0.0, 0.0, 0.0, 0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), std, statistics.median(vals), max(vals)

    s_mean, s_std, s_med, s_max = _stats([r.steps for r in records])
    t_mean, t_std, t_med, t_max = _stats([r.turns for r in records])
    firsts = [r for r in records if r.went_first]
    seconds = [r for r in records if not r.went_first]
    return EvalResult(
        opponent_label=opponent_label,
        episodes=n,
        wins=wins,
        per_deck_wins=per_deck,
        steps_mean=s_mean,
        steps_std=s_std,
        steps_median=s_med,
        steps_max=s_max,
        turns_mean=t_mean,
        turns_std=t_std,
        turns_median=t_med,
        turns_max=t_max,
        wins_first=sum(1 for r in firsts if r.win),
        episodes_first=len(firsts),
        wins_second=sum(1 for r in seconds if r.win),
        episodes_second=len(seconds),
        timeouts=sum(1 for r in records if r.timeout),
    )


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
        results.append(_aggregate_one(opp_parts, opponent_label_from_spec(spec)))
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
    env: EvalEnv,
    *,
    base_seed: int,
    episode_idx: int,
) -> _EpisodeRecord:
    """Play episode index ``episode_idx`` (1-indexed to match _episode_count).

    Reseeds the agent from ``base_seed + episode_idx`` and resets the env
    via ``reset(episode_idx=episode_idx)`` so the body is usable both by
    sequential ``run_match`` (idx=1..N in order) and by parallel-eval
    workers dispatching episodes by absolute index.

    Returns an ``_EpisodeRecord`` built from the terminal info dict.
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
    return _EpisodeRecord(
        episode_idx=episode_idx,
        win=info.get("terminal_reward", 0) > 0,
        agent_deck_idx=int(info.get("agent_deck_idx", 0)),
        steps=int(info.get("steps", 0)),
        turns=int(info.get("turn_count", 0)),
        went_first=int(info.get("agent_player", 0)) == 0,
        timeout=bool(info.get("timeout", False)),
    )


def run_match(
    agent: Opponent,
    env: EvalEnv,
    num_episodes: int,
    *,
    base_seed: int,
) -> list[_EpisodeRecord]:
    """Run ``num_episodes`` against the env's pre-configured opponent.

    Returns per-episode records in ``episode_idx`` order (1..num_episodes).
    """
    # num_episodes == 0 is a valid "skip" signal; don't pay a duel-init cost
    # just to play zero episodes (matches the pre-refactor for-loop semantics).
    if num_episodes <= 0:
        return []
    return [
        _play_one_episode(agent, env, base_seed=base_seed, episode_idx=i + 1)
        for i in range(num_episodes)
    ]


def _eval_worker(
    remote,
    *,
    agent_spec: str,
    agent_device: str,
    deck_pool: list[DeckDict],
    seed: int,
    agent_player: str,
    opponent_device: str | None,
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
) -> None:
    """Parallel-eval worker process: holds one ``EvalEnv`` cached by
    opponent_spec, plays one episode per ``("task", _EvalTask)`` message,
    replies ``("partial", _PartialResult)`` or ``("error", traceback_str)``.

    Mirrors :func:`yugioh_rl.actor_learner._actor_learner_worker`'s pipe
    protocol and shutdown handshake.
    """
    env: EvalEnv | None = None
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
                    env = EvalEnv(
                        **_make_eval_env_kwargs(
                            deck_pool,
                            task.opp_spec,
                            seed=seed,
                            agent_player=agent_player,
                            opponent_device=opponent_device,
                            deck_allocation=deck_allocation,
                            mirror_decks=mirror_decks,
                            max_steps=max_steps,
                        )
                    )
                    current_spec = task.opp_spec

                rec = _play_one_episode(
                    agent,
                    env,
                    base_seed=seed,
                    episode_idx=task.episode_idx,
                )
                remote.send(
                    (
                        _REPLY_PARTIAL,
                        _PartialResult(
                            task.opp_idx,
                            task.episode_idx,
                            rec.win,
                            rec.agent_deck_idx,
                            rec.steps,
                            rec.turns,
                            rec.went_first,
                            rec.timeout,
                        ),
                    )
                )
            except Exception:
                remote.send((_REPLY_ERROR, traceback.format_exc()))
                return
    except Exception:
        remote.send((_REPLY_ERROR, traceback.format_exc()))
    finally:
        if env is not None:
            with suppress(Exception):
                env.close()


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
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
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
                "deck_allocation": deck_allocation,
                "mirror_decks": mirror_decks,
                "max_steps": max_steps,
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
                        f"eval worker pid={proc.pid} died (exitcode={proc.exitcode}) mid-task"
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
            with suppress(BrokenPipeError, EOFError):
                worker.conn.send((_CMD_SHUTDOWN, None))
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
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Build the kwargs dict for an eval-side ``EvalEnv``.

    ``opponent_device`` is omitted when None so the ``YUGIOH_OPPONENT_DEVICE``
    env-var fallback inside ``YuGiOhEnvironment`` keeps working.
    """
    kwargs: dict[str, Any] = {
        "deck_pool": deck_pool,
        "opponent": opponent_spec,
        "seed": seed,
        "agent_player": agent_player,
        "deck_allocation": deck_allocation,
        "mirror_decks": mirror_decks,
        "max_steps": max_steps,
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
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
) -> list[EvalResult]:
    """Sequential per-opponent loop shared by both public entry points."""
    results: list[EvalResult] = []
    for spec in opponent_specs:
        env = EvalEnv(
            **_make_eval_env_kwargs(
                deck_pool,
                spec,
                seed=seed,
                agent_player=agent_player,
                opponent_device=opponent_device,
                deck_allocation=deck_allocation,
                mirror_decks=mirror_decks,
                max_steps=max_steps,
            )
        )
        try:
            records = run_match(agent, env, num_episodes, base_seed=seed)
        finally:
            env.close()
        results.append(_aggregate_one(records, opponent_label_from_spec(spec)))
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
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
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

    Sequential (``workers=1``): a fresh ``EvalEnv`` per spec with the
    same ``seed`` / ``agent_player``; ``opponent_device`` is forwarded only
    when non-None to preserve the ``YUGIOH_OPPONENT_DEVICE`` fallback.

    Parallel (``workers>=2``): episode-level results are aggregated
    deterministically — runs at different ``workers`` counts produce
    byte-equal ``EvalResult``s.
    """
    if workers <= 1:
        agent = make_eval_agent(agent_spec, seed=seed, device=agent_device)
        return _run_sequential_match_set(
            agent,
            deck_pool,
            opponent_specs,
            num_episodes=num_episodes,
            seed=seed,
            agent_player=agent_player,
            opponent_device=opponent_device,
            deck_allocation=deck_allocation,
            mirror_decks=mirror_decks,
            max_steps=max_steps,
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
        deck_allocation=deck_allocation,
        mirror_decks=mirror_decks,
        max_steps=max_steps,
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
    deck_allocation: str = "random",
    mirror_decks: bool = False,
    max_steps: int = 2000,
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
        agent,
        deck_pool,
        opponent_specs,
        num_episodes=num_episodes,
        seed=seed,
        agent_player=agent_player,
        opponent_device=opponent_device,
        deck_allocation=deck_allocation,
        mirror_decks=mirror_decks,
        max_steps=max_steps,
    )


def eval_result_to_row(r: EvalResult, deck_stems: list[str]) -> dict:
    """Normalize an EvalResult to a JSON-able row keyed by deck stem.

    Shared shape behind the eval-sweep manifest/replay and the sink-layer
    scalar flattener (``metrics_logging.flatten_eval``).
    """
    per_deck = {}
    for deck_idx, wl in r.per_deck_wins.items():
        wins = int(sum(wl))
        n = len(wl)
        per_deck[deck_stems[deck_idx]] = {
            "wins": wins,
            "episodes": n,
            "win_rate": wins / n if n else 0.0,
        }
    return {
        "win_rate": r.win_rate,
        "wins": r.wins,
        "episodes": r.episodes,
        "per_deck": per_deck,
        "steps": {
            "mean": r.steps_mean,
            "std": r.steps_std,
            "median": r.steps_median,
            "max": r.steps_max,
        },
        "turns": {
            "mean": r.turns_mean,
            "std": r.turns_std,
            "median": r.turns_median,
            "max": r.turns_max,
        },
        "play_first_rate": r.play_first_rate,
        "wins_first": r.wins_first,
        "episodes_first": r.episodes_first,
        "wins_second": r.wins_second,
        "episodes_second": r.episodes_second,
        "timeouts": r.timeouts,
    }
