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
from dataclasses import dataclass, field
from pathlib import Path

from yugioh_env.opponent import (
    NetworkOpponent,
    Opponent,
    make_opponent,
    parse_opponent_spec,
)
from yugioh_rl.env_wrapper import DeckDict, TrainingEnv

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Win-rate breakdown for one (agent, opponent) match."""

    opponent_label: str
    episodes: int
    wins: int
    win_rate: float
    per_deck_wins: dict[int, list[float]] = field(default_factory=dict)


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


def run_match(
    agent: Opponent,
    env: TrainingEnv,
    num_episodes: int,
    *,
    base_seed: int,
) -> tuple[int, dict[int, list[float]]]:
    """Run ``num_episodes`` against the env's pre-configured opponent.

    Each episode begins with an explicit ``env.reset()``; ``TrainingEnv.step()``
    no longer auto-resets.  Returns ``(total_wins, per_deck)`` where
    ``per_deck`` maps ``agent_deck_idx`` → list of 1.0/0.0 win records.
    """
    total_wins = 0
    per_deck: dict[int, list[float]] = {}
    # num_episodes == 0 is a valid "skip" signal; don't pay a duel-init cost
    # just to play zero episodes (matches the pre-refactor for-loop semantics).
    if num_episodes <= 0:
        return total_wins, per_deck
    for i in range(num_episodes):
        agent.reseed(base_seed + i + 1)
        obs = env.reset()
        done = False
        while not done:
            if agent.needs_observation:
                agent.set_observation(obs)
            action = agent.select_action(env.current_msg, env.num_actions)
            obs, reward, done, info = env.step(action)
            if done:
                win = 1.0 if info.get("terminal_reward", 0) > 0 else 0.0
                total_wins += int(win)
                deck_idx = info.get("agent_deck_idx", 0)
                per_deck.setdefault(deck_idx, []).append(win)
    return total_wins, per_deck


def evaluate(
    agent: Opponent,
    deck_pool: list[DeckDict],
    opponent_specs: list[str],
    *,
    num_episodes: int,
    seed: int,
    agent_player: str = "random",
    opponent_device: str | None = None,
) -> list[EvalResult]:
    """Run the agent against each opponent spec; return one ``EvalResult`` each.

    For each spec, a fresh ``TrainingEnv`` is built with ``reward_shaping=False``
    and the same ``seed`` / ``agent_player`` so per-opponent results are
    comparable. ``opponent_device`` is forwarded only when non-None, preserving
    the ``YUGIOH_OPPONENT_DEVICE`` env-var fallback inside ``YuGiOhEnvironment``.
    """
    results: list[EvalResult] = []
    for spec in opponent_specs:
        env_kwargs = {
            "deck_pool": deck_pool,
            "opponent": spec,
            "reward_shaping": False,
            "seed": seed,
            "agent_player": agent_player,
        }
        if opponent_device is not None:
            env_kwargs["opponent_device"] = opponent_device
        env = TrainingEnv(**env_kwargs)
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
