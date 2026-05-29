"""Coverage check: simulate many random duels for a deck and record which
cards in the decklist appeared in a SUMMONING / SPSUMMONING / FLIPSUMMONING /
CHAINING event during play.

Outputs a per-card report (✓ summoned, ✓ activated, ✗ never seen).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random as _r
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from yugioh_core.constants import (
    MSG_CHAINING,
    MSG_FLIPSUMMONING,
    MSG_SPSUMMONING,
    MSG_SUMMONING,
)
from yugioh_env import message_parser as _mp
from yugioh_env.deck_parser import parse_ydk
from yugioh_rl.env_wrapper import TrainingEnv, parse_deck_pool

SUMMON_MSGS = (MSG_SUMMONING, MSG_SPSUMMONING, MSG_FLIPSUMMONING)
ACTIVATE_MSG = MSG_CHAINING

# The only process-global state: the original `parse_messages` function,
# captured the first time we install a tap so repeated installs don't lose it.
# All counter state lives in _run_chunk's locals and travels through the
# _CoverageResult it returns.
_original_parse_messages = None


def _install_tap_for(sum_count: Counter, act_count: Counter) -> None:
    """Wrap parse_messages so every emitted message increments these counters.

    Safe to call repeatedly: the original function is captured once and the
    wrapper is rebuilt each call with the new target counters.
    """
    global _original_parse_messages
    if _original_parse_messages is None:
        _original_parse_messages = _mp.parse_messages
    original = _original_parse_messages

    def tapped(buf):
        msgs = original(buf)
        for msg in msgs:
            t = msg.get("msg_type")
            c = msg.get("code")
            if c is None or c == 0:
                continue
            if t in SUMMON_MSGS:
                sum_count[c] += 1
            elif t == ACTIVATE_MSG:
                act_count[c] += 1
        return msgs

    _mp.parse_messages = tapped
    # Also patch the already-imported reference in duel.py
    import yugioh_env.duel as _duel

    _duel.parse_messages = tapped


@dataclass
class _CoverageResult:
    """Per-worker snapshot of the tap state, returned to the parent for merging.

    `sum_count.keys()` is the "summoned at least once" set; same for `act_count`.
    """

    sum_count: Counter[int] = field(default_factory=Counter)
    act_count: Counter[int] = field(default_factory=Counter)


def _split_episodes(total: int, workers: int) -> list[tuple[int, int]]:
    """Partition [0, total) into `workers` contiguous ranges, distributing remainder."""
    base, rem = divmod(total, workers)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for w in range(workers):
        size = base + (1 if w < rem else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def _run_chunk_to_pipe(
    remote,
    deck_path: str,
    base_seed: int,
    start: int,
    end: int,
) -> None:
    """Worker entry point: run [start, end) and send a _CoverageResult through the pipe.

    Using a Pipe-based handoff (matches yugioh_rl/eval.py) avoids
    multiprocessing.Pool's SemLock requirement, which is blocked under
    sandboxed environments on macOS.
    """
    try:
        result = _run_chunk((deck_path, base_seed, start, end))
        remote.send(result)
    finally:
        remote.close()


def _run_chunk(args: tuple[str, int, int, int]) -> _CoverageResult:
    """Run episodes [start, end) and return a snapshot of the tap state.

    Each episode uses `episode_idx == base_seed + ep_idx` for the env's
    deck/seed sampling and the same value seeds the action RNG, so results
    are reproducible per-episode regardless of how the work is partitioned.
    """
    deck_path, base_seed, start, end = args

    sum_count: Counter[int] = Counter()
    act_count: Counter[int] = Counter()
    _install_tap_for(sum_count, act_count)

    deck_pool = parse_deck_pool([deck_path, deck_path])
    env = TrainingEnv(
        deck_pool=deck_pool,
        opponent="random",
        seed=base_seed,
        agent_player="random",
        reward_shaping=False,
    )

    for ep_idx in range(start, end):
        obs = env.reset(episode_idx=ep_idx)
        rng = _r.Random(base_seed + ep_idx)
        done = False
        while not done:
            mask = obs["action_mask"]
            legal = [i for i, x in enumerate(mask) if x]
            if not legal:
                break
            action = rng.choice(legal)
            obs, _reward, done, _info = env.step(action)
        if (ep_idx - start + 1) % 50 == 0:
            print(
                f"  worker[{start}-{end}) episode {ep_idx + 1}",
                file=sys.stderr,
            )

    return _CoverageResult(sum_count=sum_count, act_count=act_count)


def _merge(results: list[_CoverageResult]) -> _CoverageResult:
    merged = _CoverageResult()
    for r in results:
        merged.sum_count += r.sum_count
        merged.act_count += r.act_count
    return merged


def lookup_card_info(card_ids: list[int]) -> dict[int, tuple[str, int]]:
    """Return {id: (name, type_flags)} for each card."""
    # Resolve cards.cdb relative to the repo root, not the CWD, so the
    # harness works from any directory.
    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "assets" / "cards.cdb"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    placeholders = ",".join("?" * len(card_ids))
    rows = cur.execute(
        f"SELECT datas.id, texts.name, datas.type "
        f"FROM datas JOIN texts USING(id) WHERE datas.id IN ({placeholders})",
        card_ids,
    ).fetchall()
    conn.close()
    return {rid: (name, type_flags) for rid, name, type_flags in rows}


# ygopro type-flag bits (from constants/cardscripts)
TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4


def kind_of(type_flags: int) -> str:
    if type_flags & TYPE_MONSTER:
        return "monster"
    if type_flags & TYPE_TRAP:
        return "trap"
    if type_flags & TYPE_SPELL:
        return "spell"
    return "?"


def run_coverage(deck_path: str, episodes: int, seed: int, workers: int = 1) -> int:
    deck = parse_ydk(deck_path)
    deck_card_ids = sorted(set(deck["main"] + deck["extra"]))
    info = lookup_card_info(deck_card_ids)

    if workers <= 1:
        # Single-process path: run in-process, no Pool overhead.
        result = _run_chunk((deck_path, seed, 0, episodes))
    else:
        ranges = _split_episodes(episodes, workers)
        # Drop empty ranges (e.g. workers > episodes).
        chunks = [(s, e) for s, e in ranges if e > s]
        ctx = mp.get_context("spawn")
        procs: list[tuple[mp.process.BaseProcess, mp.connection.Connection]] = []
        for start, end in chunks:
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_run_chunk_to_pipe,
                args=(child_conn, deck_path, seed, start, end),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            procs.append((proc, parent_conn))

        partials: list[_CoverageResult] = []
        try:
            for proc, conn in procs:
                partials.append(conn.recv())
                conn.close()
                proc.join()
        finally:
            for proc, _ in procs:
                if proc.is_alive():
                    proc.terminate()
        result = _merge(partials)

    # Report
    unique_main = sorted(set(deck["main"]))
    unique_extra = sorted(set(deck["extra"]))

    def report(section: str, cards: list[int]) -> None:
        print(f"\n=== {section} ({len(cards)} unique cards) ===")
        print(f"{'kind':>8}  {'summoned':>10}  {'activated':>10}  {'id':>9}  name")
        for cid in cards:
            name, type_flags = info.get(cid, ("<unknown>", 0))
            k = kind_of(type_flags)
            # Spells/traps are never "summoned"; show "/" instead of MISS 0
            # so the column conveys "not applicable" rather than "missing".
            if k == "monster":
                s_mark = "OK" if cid in result.sum_count else "MISS"
                s_cell = f"{s_mark:<4} {result.sum_count.get(cid, 0):>5}"
            else:
                s_cell = "/"
            a_mark = "OK" if cid in result.act_count else "MISS"
            a_cell = f"{a_mark:<4} {result.act_count.get(cid, 0):>5}"
            print(f"{k:>8}  {s_cell:>10}  {a_cell:>10}  {cid:>9}  {name}")

    report("MAIN", unique_main)
    report("EXTRA", unique_extra)

    # Aggregate issues
    monster_ids = [c for c in deck_card_ids if kind_of(info.get(c, ("", 0))[1]) == "monster"]
    spell_trap_ids = [
        c for c in deck_card_ids if kind_of(info.get(c, ("", 0))[1]) in ("spell", "trap")
    ]

    # A monster is "dead" only if it's neither summoned nor activated.
    # Hand traps (Maxx "C", Ash, Veiler) and cards activated from deck/GY
    # (Dotscaper, Elder Entity N'tss as material) often fire MSG_CHAINING
    # without ever being summoned, which proves they're playing.
    dead_monsters = [
        c for c in monster_ids if c not in result.sum_count and c not in result.act_count
    ]
    monsters_no_activation_seen = [
        c for c in monster_ids if (c in result.sum_count) and (c not in result.act_count)
    ]
    st_never_activated = [c for c in spell_trap_ids if c not in result.act_count]

    print("\n=== Issues ===")
    issue_count = 0
    if dead_monsters:
        issue_count += len(dead_monsters)
        print(f"  Monsters never summoned and never activated ({len(dead_monsters)}):")
        for cid in dead_monsters:
            print(f"    {cid}  {info.get(cid, ('?',))[0]}")
    if monsters_no_activation_seen:
        # Not all monsters have activatable effects (e.g. vanillas, beatsticks).
        # Just list as informational.
        print(
            f"  Monsters summoned but no effect activation observed "
            f"({len(monsters_no_activation_seen)}) — informational, may be vanilla:"
        )
        for cid in monsters_no_activation_seen:
            print(f"    {cid}  {info.get(cid, ('?',))[0]}")
    if st_never_activated:
        issue_count += len(st_never_activated)
        print(f"  Spells/Traps never activated ({len(st_never_activated)}):")
        for cid in st_never_activated:
            print(f"    {cid}  {info.get(cid, ('?',))[0]}")
    if not (dead_monsters or st_never_activated):
        print("  No hard issues: every monster summoned or activated, every spell/trap activated.")

    return 0 if issue_count == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, single-process).",
    )
    args = ap.parse_args()
    return run_coverage(args.deck, args.episodes, args.seed, workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
