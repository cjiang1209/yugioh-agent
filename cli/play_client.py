#!/usr/bin/env python3
"""Interactive client for the Yu-Gi-Oh! environment server.

Usage:
    # Start the server first:
    #   uvicorn yugioh_env.server.app:app --host 0.0.0.0 --port 8000

    # Then run this client:
    python scripts/play_client.py                    # interactive mode (default)
    python scripts/play_client.py --mode random      # random agent
    python scripts/play_client.py --mode greedy      # greedy agent (first legal action)
    python scripts/play_client.py --url ws://host:8000
    python scripts/play_client.py --episodes 10 --mode random
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

from yugioh_env.client import YuGiOhEnv
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.models import YuGiOhAction, YuGiOhObservation

# Phase ID -> human-readable name
PHASE_NAMES = {
    0x01: "Draw",
    0x02: "Standby",
    0x04: "Main 1",
    0x08: "Battle Start",
    0x10: "Battle Step",
    0x20: "Damage",
    0x40: "Damage Calc",
    0x80: "Battle",
    0x100: "Main 2",
    0x200: "End",
}

# MSG_SELECT type -> human-readable name
MSG_SELECT_NAMES = {
    10: "Battle Command",
    11: "Idle Command",
    12: "Effect Yes/No",
    13: "Yes/No",
    14: "Option",
    15: "Select Card",
    16: "Chain",
    18: "Select Place",
    19: "Select Position",
    20: "Tribute",
    22: "Select Counter",
    23: "Select Sum",
    24: "Select Dis-Field",
    26: "Select/Unselect Card",
}

# Idle command categories
IDLE_CATEGORIES = {
    0: "Normal Summon",
    1: "Special Summon",
    2: "Reposition",
    3: "Monster Set",
    4: "Spell/Trap Set",
    5: "Activate Effect",
    6: "-> Battle Phase",
    7: "-> End Phase",
}

# Battle command categories
BATTLE_CATEGORIES = {
    0: "Activate Effect",
    1: "Attack",
    2: "-> Main Phase 2",
    3: "-> End Phase",
}


class CardNames:
    """Card name lookup from cards.cdb, preloaded at startup."""

    def __init__(self):
        self._names: dict[int, str] = {}

    def load(self, card_codes: set[int] | None = None) -> None:
        """Load card names from cards.cdb.

        Args:
            card_codes: If provided, only load names for these codes.
                        If None, load all card names.
        """
        db_path = os.environ.get("YUGIOH_DB_PATH")
        if not db_path:
            db_path = str(Path(__file__).resolve().parent.parent / "assets" / "cards.cdb")
        if not os.path.isfile(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            if card_codes is not None:
                placeholders = ",".join("?" for _ in card_codes)
                rows = conn.execute(
                    f"SELECT id, name FROM texts WHERE id IN ({placeholders})",
                    list(card_codes),
                )
            else:
                rows = conn.execute("SELECT id, name FROM texts")
            self._names = {row[0]: row[1] for row in rows}
        finally:
            conn.close()

    def get(self, code: int) -> str | None:
        return self._names.get(code)


card_names = CardNames()


def decode_u16_le(lo: int, hi: int) -> int:
    """Decode two uint8 bytes (little-endian) into a uint16."""
    return lo | (hi << 8)


def parse_global_state(gs: list[int]) -> dict:
    """Parse global_state vector into human-readable dict."""
    return {
        "my_lp": decode_u16_le(gs[0], gs[1]),
        "opp_lp": decode_u16_le(gs[2], gs[3]),
        "turn": gs[4],
        "phase": gs[5],
        "is_my_turn": bool(gs[6]),
        "chain_count": gs[7],
        "msg_type": gs[8],
        "my_deck": gs[9],
        "my_hand": gs[10],
        "my_grave": gs[11],
        "my_banished": gs[12],
        "my_extra": gs[13],
        "opp_deck": gs[14],
        "opp_hand": gs[15],
        "opp_grave": gs[16],
        "opp_banished": gs[17],
        "opp_extra": gs[18],
        "is_finished": bool(gs[19]),
    }


def decode_u32_le(b0: int, b1: int, b2: int, b3: int) -> int:
    """Decode four uint8 bytes (little-endian) into a uint32."""
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def parse_action_features(action_feats: list[int]) -> dict:
    """Parse a single action's feature vector."""
    msg_type = action_feats[0]
    category = action_feats[1]
    code = decode_u32_le(action_feats[2], action_feats[3], action_feats[4], action_feats[5])
    location = action_feats[6]
    sequence = action_feats[7]
    index = action_feats[8]
    num_selected = action_feats[9] if action_feats[9] > 0 else 1
    extra_indices = []
    if num_selected >= 2:
        extra_indices.append(action_feats[10])
    if num_selected >= 3:
        extra_indices.append(action_feats[11])
    return {
        "msg_type": msg_type,
        "category": category,
        "code": code,
        "location": location,
        "sequence": sequence,
        "index": index,
        "num_selected": num_selected,
        "extra_indices": extra_indices,
    }


def describe_action(action_feats: list[int]) -> str:
    """Produce a human-readable description of an action."""
    info = parse_action_features(action_feats)
    msg_type = info["msg_type"]
    cat = info["category"]
    code = info["code"]

    parts = []

    if msg_type == 11:  # IDLE
        cat_name = IDLE_CATEGORIES.get(cat, f"cat={cat}")
        parts.append(cat_name)
    elif msg_type == 10:  # BATTLE
        cat_name = BATTLE_CATEGORIES.get(cat, f"cat={cat}")
        parts.append(cat_name)
    elif msg_type == 12:  # EFFECTYN
        parts.append("Yes" if cat == 0 else "No")
    elif msg_type == 13:  # YESNO
        parts.append("Yes" if cat == 0 else "No")
    elif msg_type == 19:  # POSITION
        pos_names = {0x1: "FU-ATK", 0x2: "FD-ATK", 0x4: "FU-DEF", 0x8: "FD-DEF"}
        parts.append(f"Position: {pos_names.get(info['index'], info['index'])}")
    elif msg_type in (18, 24):  # SELECT_PLACE / SELECT_DISFIELD
        sel_name = MSG_SELECT_NAMES.get(msg_type, f"msg={msg_type}")
        loc = info["location"]
        zone_name = "Monster" if loc == 0x04 else "Spell/Trap" if loc == 0x08 else f"loc=0x{loc:02x}"
        parts.append(f"{sel_name} — {zone_name} Zone {info['sequence']}")
    elif msg_type == 15 and cat == 1:  # SELECT_CARD finish
        num_sel = info.get("num_selected", 0)
        parts.append(f"Finish selecting ({num_sel} card{'s' if num_sel != 1 else ''})")
    else:
        sel_name = MSG_SELECT_NAMES.get(msg_type, f"msg={msg_type}")
        num_sel = info.get("num_selected", 1)
        if num_sel > 1:
            idx_strs = [f"#{info['index']}"] + [f"#{ei}" for ei in info.get("extra_indices", [])]
            parts.append(f"{sel_name} {'+'.join(idx_strs)}")
        else:
            parts.append(f"{sel_name} #{info['index']}")

    if code > 0:
        name = card_names.get(code)
        if name:
            parts.append(f"({code}: {name})")
        else:
            parts.append(f"({code})")

    return " ".join(parts)


def display_state(obs: YuGiOhObservation, step_num: int) -> None:
    """Print a summary of the current observation."""
    gs = parse_global_state(obs.global_state)
    phase_name = PHASE_NAMES.get(gs["phase"], f"0x{gs['phase']:02x}")
    turn_marker = " <-- YOUR TURN" if gs["is_my_turn"] else ""

    print()
    print(f"{'=' * 60}")
    print(f"  Step {step_num}  |  Turn {gs['turn']}  |  Phase: {phase_name}{turn_marker}")
    print(f"{'=' * 60}")
    print(f"  YOUR LP: {gs['my_lp']:>5}    |  OPP LP: {gs['opp_lp']:>5}")
    print(f"  Hand: {gs['my_hand']:>2}  Deck: {gs['my_deck']:>2}  GY: {gs['my_grave']:>2}  "
          f"Ban: {gs['my_banished']:>2}  Extra: {gs['my_extra']:>2}")
    print(f"  Opp Hand: {gs['opp_hand']:>2}  Deck: {gs['opp_deck']:>2}  GY: {gs['opp_grave']:>2}  "
          f"Ban: {gs['opp_banished']:>2}  Extra: {gs['opp_extra']:>2}")

    if gs["chain_count"] > 0:
        print(f"  Chain count: {gs['chain_count']}")

    # Show legal actions
    legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
    msg_type = gs["msg_type"]
    msg_name = MSG_SELECT_NAMES.get(msg_type, f"msg={msg_type}")

    print(f"{'─' * 60}")
    print(f"  Decision: {msg_name}  ({len(legal)} legal action{'s' if len(legal) != 1 else ''})")
    print(f"{'─' * 60}")

    for idx in legal:
        desc = describe_action(obs.actions[idx])
        print(f"    [{idx:>2}]  {desc}")

    print()


def pick_action_interactive(obs: YuGiOhObservation) -> int:
    """Prompt the user to pick a legal action."""
    legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
    if len(legal) == 1:
        print(f"  >> Auto-selecting only legal action: [{legal[0]}]")
        return legal[0]

    while True:
        try:
            raw = input("  >> Enter action index (or 'q' to quit): ").strip()
            if raw.lower() == "q":
                raise KeyboardInterrupt
            choice = int(raw)
            if choice in legal:
                return choice
            print(f"     Invalid: {choice} is not a legal action. Legal: {legal}")
        except ValueError:
            print("     Please enter a number.")


def pick_action_random(obs: YuGiOhObservation) -> int:
    """Pick a random legal action."""
    legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
    return random.choice(legal)


def pick_action_greedy(obs: YuGiOhObservation) -> int:
    """Pick the first legal action (greedy/deterministic)."""
    for i, m in enumerate(obs.action_mask):
        if m == 1:
            return i
    return 0


def run_episode(
    env: YuGiOhEnv,
    pick_action,
    seed: int | None = None,
    verbose: bool = True,
    deck0: dict | None = None,
    deck1: dict | None = None,
) -> dict:
    """Run a single duel episode. Returns stats dict."""
    reset_kwargs = {}
    if seed is not None:
        reset_kwargs["seed"] = seed
    if deck0 is not None:
        reset_kwargs["deck0"] = deck0
    if deck1 is not None:
        reset_kwargs["deck1"] = deck1

    result = env.reset(**reset_kwargs)
    step_num = 0

    if verbose:
        display_state(result.observation, step_num)

    while not result.done:
        action_idx = pick_action(result.observation)
        if verbose:
            desc = describe_action(result.observation.actions[action_idx])
            print(f"  -> Playing action [{action_idx}]: {desc}")

        result = env.step(YuGiOhAction(action_index=action_idx))
        step_num += 1

        if verbose and not result.done:
            display_state(result.observation, step_num)

    # Final summary
    gs = parse_global_state(result.observation.global_state)
    reward = result.reward

    if verbose:
        print()
        print(f"{'#' * 60}")
        print(f"  DUEL OVER after {step_num} steps")
        print(f"  Final LP — You: {gs['my_lp']}  |  Opponent: {gs['opp_lp']}")
        print(f"  Reward: {reward}")
        if reward is not None and reward > 0:
            print("  Result: WIN!")
        elif reward is not None and reward < 0:
            print("  Result: LOSS")
        else:
            print("  Result: DRAW")
        print(f"{'#' * 60}")
        print()

    return {
        "steps": step_num,
        "reward": reward,
        "my_lp": gs["my_lp"],
        "opp_lp": gs["opp_lp"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Yu-Gi-Oh! RL environment client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url", default="ws://localhost:8000",
        help="Server URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--mode", choices=["interactive", "random", "greedy"], default="interactive",
        help="Play mode (default: interactive)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to play (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the first episode (incremented per episode)",
    )
    parser.add_argument(
        "--deck", type=str, default=None,
        help="Path to .ydk deck file (used for both players)",
    )
    parser.add_argument(
        "--deck0", type=str, default=None,
        help="Path to .ydk deck file for player 0 (overrides --deck)",
    )
    parser.add_argument(
        "--deck1", type=str, default=None,
        help="Path to .ydk deck file for player 1 (overrides --deck)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output (show only episode summaries)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Message timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    # Seed client-side RNG for reproducible random-mode play
    if args.seed is not None:
        random.seed(args.seed)

    pickers = {
        "interactive": pick_action_interactive,
        "random": pick_action_random,
        "greedy": pick_action_greedy,
    }
    pick_action = pickers[args.mode]
    verbose = not args.quiet

    # Parse deck files into inline card-code dicts
    deck0 = None
    deck1 = None
    if args.deck:
        shared_deck = parse_ydk(args.deck)
        deck0 = shared_deck
        deck1 = shared_deck
    if args.deck0:
        deck0 = parse_ydk(args.deck0)
    if args.deck1:
        deck1 = parse_ydk(args.deck1)

    # Collect card codes from specified decks to limit DB loading
    deck_codes: set[int] | None = None
    if deck0 is not None or deck1 is not None:
        deck_codes = set()
        for d in (deck0, deck1):
            if d is not None:
                deck_codes.update(d.get("main", []))
                deck_codes.update(d.get("extra", []))

    card_names.load(deck_codes)
    print(f"Loaded {len(card_names._names)} card names.")
    print(f"Connecting to {args.url} ...")

    try:
        with YuGiOhEnv(
            base_url=args.url,
            message_timeout_s=args.timeout,
        ) as env:
            print("Connected!\n")

            all_stats = []
            for ep in range(args.episodes):
                seed = args.seed + ep if args.seed is not None else None
                if args.episodes > 1:
                    print(f"--- Episode {ep + 1}/{args.episodes} (seed={seed}) ---")

                t0 = time.time()
                stats = run_episode(env, pick_action, seed=seed, verbose=verbose,
                                    deck0=deck0, deck1=deck1)
                elapsed = time.time() - t0
                stats["time"] = elapsed
                all_stats.append(stats)

                if args.episodes > 1:
                    r = stats["reward"]
                    tag = "WIN" if r is not None and r > 0 else ("LOSS" if r is not None and r < 0 else "DRAW")
                    print(f"  => {tag} in {stats['steps']} steps, {elapsed:.1f}s\n")

            # Print aggregate stats for multi-episode runs
            if args.episodes > 1:
                wins = sum(1 for s in all_stats if s["reward"] is not None and s["reward"] > 0)
                losses = sum(1 for s in all_stats if s["reward"] is not None and s["reward"] < 0)
                draws = args.episodes - wins - losses
                avg_steps = sum(s["steps"] for s in all_stats) / len(all_stats)
                total_time = sum(s["time"] for s in all_stats)

                print(f"{'=' * 60}")
                print(f"  {args.episodes} episodes complete")
                print(f"  Wins: {wins}  Losses: {losses}  Draws: {draws}  "
                      f"Win rate: {wins / args.episodes:.1%}")
                print(f"  Avg steps: {avg_steps:.1f}  Total time: {total_time:.1f}s")
                print(f"{'=' * 60}")

    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Is the server running? Start it with:", file=sys.stderr)
        print("  uvicorn yugioh_env.server.app:app --host 0.0.0.0 --port 8000", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
