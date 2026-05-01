"""On-demand head-to-head match between two leaderboard entries.

Same eval primitive as ``score.py``, but the opponent is
``model:<entry_b's checkpoint>`` rather than the panel. Results are
mirrored symmetrically into both entries' ``pairwise_results`` lists
(A's record says "vs B = 58%", B's says "vs A = 42%").

Assumes the engine never returns ties: ``wins + losses == episodes``
for every match. The mirror logic depends on this — if it ever fails,
B's win counts go negative.

Two ``write_entry`` calls in sequence are NOT transactionally paired;
the leaderboard is briefly inconsistent if the second write fails. This
is acceptable under the single-user / single-process CLI model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from yugioh_leaderboard.entry import (
    Entry,
    PairwiseMatchResult,
    deck_summary,
    now_iso,
    stable_seed,
)
from yugioh_leaderboard.panel import PanelConfig


class NoSharedDecksError(ValueError):
    """Raised when two entries' deck pools have no overlap and no override is given."""


def _pair_seed(eid_a: str, eid_b: str) -> int:
    """Order-independent seed so ``pairwise A B`` and ``pairwise B A``
    re-run the same match rather than two independently-seeded ones."""
    lo, hi = sorted((eid_a, eid_b))
    return stable_seed("pair", lo, hi)


def replace_or_append_pairwise(
    results: list[PairwiseMatchResult], new: PairwiseMatchResult
) -> list[PairwiseMatchResult]:
    """Drop any prior record matching ``new.vs_entry_id`` and append ``new``.

    Re-running a pairwise match overwrites the prior record rather than
    duplicating it, keeping the leaderboard a single source of truth.
    """
    out = [r for r in results if r.vs_entry_id != new.vs_entry_id]
    out.append(new)
    return out


def run_pairwise(
    entry_a: Entry,
    entry_b: Entry,
    panel: PanelConfig,
    *,
    episodes: int = 100,
    seed: Optional[int] = None,
    decks_override: Optional[list[str]] = None,
) -> tuple[Entry, Entry]:
    """Run one pairwise match and return updated copies of (entry_a, entry_b).

    Decks default to the intersection of both entries' deck pools (or
    ``decks_override`` if passed). Seed defaults to a stable hash of the
    two entry ids.
    """
    from cli.utils import resolve_device
    from yugioh_rl.env_wrapper import parse_deck_pool
    from yugioh_rl.eval import evaluate, make_eval_agent

    base_seed = seed if seed is not None else _pair_seed(entry_a.entry_id, entry_b.entry_id)
    device = resolve_device(panel.match.device)

    if decks_override is not None:
        decks = decks_override
    else:
        a_decks = list(entry_a.features.get("deck_paths") or [])
        b_decks = list(entry_b.features.get("deck_paths") or [])
        decks = [p for p in a_decks if p in b_decks]
        if not decks:
            raise NoSharedDecksError(
                f"entries {entry_a.entry_id} and {entry_b.entry_id} share no decks; "
                "pass --decks to override"
            )

    deck_pool = parse_deck_pool(decks)
    deck_stems = [Path(p).stem for p in decks]

    agent = make_eval_agent(
        f"model:{entry_a.checkpoint_path}", seed=base_seed, device=device
    )
    raw = evaluate(
        agent,
        deck_pool=deck_pool,
        opponent_specs=[f"model:{entry_b.checkpoint_path}"],
        num_episodes=episodes,
        seed=base_seed,
        agent_player=panel.match.agent_player,
        opponent_device=device,
    )
    r = raw[0]
    if r.wins > r.episodes:
        raise AssertionError(
            f"engine reported wins ({r.wins}) > episodes ({r.episodes}); "
            "pairwise mirror invariant broken"
        )

    per_deck_a = {
        deck_stems[deck_idx]: deck_summary(int(sum(wl)), len(wl))
        for deck_idx, wl in r.per_deck_wins.items()
    }
    per_deck_b = {
        name: deck_summary(d["episodes"] - d["wins"], d["episodes"])
        for name, d in per_deck_a.items()
    }
    b_wins = r.episodes - r.wins
    a_record = PairwiseMatchResult(
        vs_entry_id=entry_b.entry_id,
        vs_checkpoint_hash=entry_b.checkpoint_hash,
        episodes=r.episodes, wins=r.wins, win_rate=r.win_rate,
        per_deck=per_deck_a, seed=base_seed, evaluated_at=now_iso(),
    )
    b_record = PairwiseMatchResult(
        vs_entry_id=entry_a.entry_id,
        vs_checkpoint_hash=entry_a.checkpoint_hash,
        episodes=r.episodes, wins=b_wins,
        win_rate=b_wins / r.episodes if r.episodes else 0.0,
        per_deck=per_deck_b, seed=base_seed, evaluated_at=now_iso(),
    )

    return (
        replace(entry_a, pairwise_results=replace_or_append_pairwise(
            entry_a.pairwise_results, a_record)),
        replace(entry_b, pairwise_results=replace_or_append_pairwise(
            entry_b.pairwise_results, b_record)),
    )
