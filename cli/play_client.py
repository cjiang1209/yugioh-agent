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
import sys
import time
from pathlib import Path

from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import PHASE_NAMES
from yugioh_core.string_resolver import parse_sys_strings
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.client import YuGiOhEnv
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.models import YuGiOhAction, YuGiOhObservation


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


def display_events(event_log: list[str]) -> None:
    """Print event log with prominent formatting."""
    if not event_log:
        return
    print()
    for event in event_log:
        print(f"\033[36m  >> {event}\033[0m")
    print()


_PROMPT_TYPE_LABELS = {
    "idle_cmd": "Idle Command",
    "battle_cmd": "Battle Command",
    "effect_yn": "Effect Yes/No",
    "yes_no": "Yes/No",
    "option": "Option",
    "select_card": "Select Card",
    "chain_link": "Chain Link",
    "place": "Place",
    "position": "Position",
    "tribute": "Tribute",
    "sort_card": "Sort Cards",
    "number": "Announce Number",
    "race": "Announce Race",
    "attribute": "Announce Attribute",
    "rps": "Rock-Paper-Scissors",
    "counter": "Select Counter",
    "unknown": "Unknown",
}


def _format_prompt_summary(prompt: dict | None) -> str:
    """Render a one-line summary of the prompt's type and key constraints.

    Returns just the type label for prompts whose action labels already
    carry the full information (yes/no, position, etc.); adds min/max
    constraints and selection progress for the few prompts that need it
    (select_card, tribute, chain).
    """
    if not prompt:
        return ""
    p_type = prompt.get("type", "unknown")
    label = _PROMPT_TYPE_LABELS.get(p_type, p_type)

    if p_type in ("yes_no", "effect_yn"):
        text = prompt.get("prompt_text")
        return f"{label} — {text}" if text else label

    if p_type == "select_card":
        lo, hi = prompt["min"], prompt["max"]
        range_str = f"pick {lo}" if lo == hi else f"pick {lo} to {hi}"
        # selected_count is set for MSG_SELECT_CARD only; MSG_SELECT_UNSELECT_CARD
        # emits `finishable` instead and omits selected_count.
        picked = prompt.get("selected_count", 0)
        progress = f", {picked} selected" if picked else ""
        finishable = ", finishable" if prompt.get("finishable") else ""
        return f"{label} — {range_str}{progress}{finishable}"

    if p_type == "tribute":
        min_rel = prompt["min_release"]
        max_cards = prompt["max_cards"]
        rel_total = prompt["release_total"]
        picked = prompt["cards_selected"]
        progress = (
            f", release={rel_total}/{min_rel} ({picked} card{'s' if picked != 1 else ''})"
            if picked
            else ""
        )
        return f"{label} — release total ≥ {min_rel} (max {max_cards} cards){progress}"

    if p_type == "chain_link" and prompt.get("forced"):
        return f"{label} — forced"

    return label


def display_state(obs: YuGiOhObservation, step_num: int, describer: ActionDescriber) -> None:
    """Print a summary of the current observation."""
    gs = parse_global_state(obs.global_state)
    phase_name = PHASE_NAMES.get(gs["phase"], f"0x{gs['phase']:02x}")
    turn_marker = " <-- YOUR TURN" if gs["is_my_turn"] else ""

    print()
    print(f"{'=' * 60}")
    print(f"  Step {step_num}  |  Turn {gs['turn']}  |  Phase: {phase_name}{turn_marker}")
    print(f"{'=' * 60}")
    print(f"  YOUR LP: {gs['my_lp']:>5}    |  OPP LP: {gs['opp_lp']:>5}")
    print(
        f"  Hand: {gs['my_hand']:>2}  Deck: {gs['my_deck']:>2}  GY: {gs['my_grave']:>2}  "
        f"Ban: {gs['my_banished']:>2}  Extra: {gs['my_extra']:>2}"
    )
    print(
        f"  Opp Hand: {gs['opp_hand']:>2}  Deck: {gs['opp_deck']:>2}  GY: {gs['opp_grave']:>2}  "
        f"Ban: {gs['opp_banished']:>2}  Extra: {gs['opp_extra']:>2}"
    )

    if gs["chain_count"] > 0:
        print(f"  Chain count: {gs['chain_count']}")

    legal_count = sum(1 for m in obs.action_mask if m == 1)
    summary = _format_prompt_summary(describer.describe_prompt(obs))
    decision_line = (
        f"  Decision: {summary}  ({legal_count} legal action{'s' if legal_count != 1 else ''})"
        if summary
        else f"  Decision: {legal_count} legal action{'s' if legal_count != 1 else ''}"
    )

    print(f"{'─' * 60}")
    print(decision_line)
    print(f"{'─' * 60}")

    for d in describer.describe_all(obs):
        print(f"    [{d.index:>2}]  {d.description}")

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
    describer: ActionDescriber,
    seed: int | None = None,
    verbose: bool = True,
    deck0: dict | None = None,
    deck1: dict | None = None,
    agent_player: int | None = None,
) -> dict:
    """Run a single duel episode. Returns stats dict."""
    reset_kwargs = {}
    if seed is not None:
        reset_kwargs["seed"] = seed
    if deck0 is not None:
        reset_kwargs["deck0"] = deck0
    if deck1 is not None:
        reset_kwargs["deck1"] = deck1
    if agent_player is not None:
        reset_kwargs["agent_player"] = agent_player

    result = env.reset(**reset_kwargs)
    step_num = 0

    if verbose:
        display_events(result.observation.event_log)
        display_state(result.observation, step_num, describer)

    while not result.done:
        action_idx = pick_action(result.observation)
        if verbose:
            d = describer.describe(result.observation, action_idx)
            print(f"  -> Playing action [{action_idx}]: {d.description}")

        result = env.step(YuGiOhAction(action_index=action_idx))
        step_num += 1

        if verbose and not result.done:
            display_events(result.observation.event_log)
            display_state(result.observation, step_num, describer)

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
        "--url",
        default="ws://localhost:8000",
        help="Server URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "random", "greedy"],
        default="interactive",
        help="Play mode (default: interactive)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the first episode (incremented per episode)",
    )
    parser.add_argument(
        "--deck",
        type=str,
        default=None,
        help="Path to .ydk deck file (used for both players)",
    )
    parser.add_argument(
        "--deck0",
        type=str,
        default=None,
        help="Path to .ydk deck file for player 0 (overrides --deck)",
    )
    parser.add_argument(
        "--deck1",
        type=str,
        default=None,
        help="Path to .ydk deck file for player 1 (overrides --deck)",
    )
    player_order = parser.add_mutually_exclusive_group()
    player_order.add_argument(
        "--go-first",
        action="store_true",
        default=True,
        help="Agent goes first (player 0, default)",
    )
    player_order.add_argument(
        "--go-second",
        action="store_true",
        help="Agent goes second (player 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output (show only episode summaries)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
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

    agent_player = 1 if args.go_second else 0

    db_path = os.environ.get("YUGIOH_DB_PATH") or str(
        Path(__file__).resolve().parent.parent / "assets" / "cards.cdb"
    )
    strings_path = os.environ.get("YUGIOH_STRINGS_PATH") or str(
        Path(__file__).resolve().parent.parent / "assets" / "strings.conf"
    )
    card_db = CardDatabase(db_path)
    sys_strings = parse_sys_strings(strings_path) if Path(strings_path).is_file() else None
    describer = ActionDescriber(card_db, sys_strings=sys_strings)
    print(f"Agent player: {agent_player} ({'goes second' if agent_player == 1 else 'goes first'})")
    print(f"Connecting to {args.url} ...")

    try:
        with YuGiOhEnv(
            base_url=args.url,
            message_timeout_s=args.timeout,
        ).sync() as env:
            print("Connected!\n")

            all_stats = []
            for ep in range(args.episodes):
                seed = args.seed + ep if args.seed is not None else None
                if args.episodes > 1:
                    print(f"--- Episode {ep + 1}/{args.episodes} (seed={seed}) ---")

                t0 = time.time()
                stats = run_episode(
                    env,
                    pick_action,
                    describer,
                    seed=seed,
                    verbose=verbose,
                    deck0=deck0,
                    deck1=deck1,
                    agent_player=agent_player,
                )
                elapsed = time.time() - t0
                stats["time"] = elapsed
                all_stats.append(stats)

                if args.episodes > 1:
                    r = stats["reward"]
                    tag = (
                        "WIN"
                        if r is not None and r > 0
                        else ("LOSS" if r is not None and r < 0 else "DRAW")
                    )
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
                print(
                    f"  Wins: {wins}  Losses: {losses}  Draws: {draws}  "
                    f"Win rate: {wins / args.episodes:.1%}"
                )
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
