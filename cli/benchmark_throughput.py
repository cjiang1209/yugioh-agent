#!/usr/bin/env python3
"""Benchmark cli.train throughput across a (num_envs × rollout_steps) grid.

Each combination runs a short training session, log lines like
``Update X/Y | steps=N | FPS=...`` are parsed, and steady-state FPS is computed
from late-window timestamp deltas (so per-process startup is excluded).

Defaults isolate environment/rollout throughput from other costs:
  - opponent=random  (no model inference in the env step)
  - rnn_type=none    (no recurrent state plumbing)
  - no --card-embeddings (symbolic encoder)

Outputs a 2D FPS table to stdout, picks the best combination, and persists a
JSON summary alongside per-run logs under ``benchmarks/<id>/``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cli.utils import DEVICE_CHOICES
from yugioh_rl.config import VEC_ENV_TYPES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "benchmarks"

LOG_PAT = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}).*Update (\d+)/(\d+) \| steps=(\d+).*FPS=([\d.]+)"
)


def parse_log(path: Path) -> dict | None:
    rows = []
    for line in path.read_text().splitlines():
        m = LOG_PAT.search(line)
        if m:
            h, mi, s, upd, _total, steps, _fps = m.groups()
            t = int(h) * 3600 + int(mi) * 60 + int(s)
            rows.append((int(upd), int(steps), t))
    row_count = len(rows)
    if row_count < 2:
        return None

    # Skip warmup rows but keep >=2 in the window so steps/dt is non-degenerate.
    skip = 0 if row_count < 3 else min(max(1, row_count // 4), row_count - 2)
    first, last = rows[skip], rows[-1]

    dt = (last[2] - first[2]) % 86400  # midnight wrap guard
    if dt <= 0:
        return None
    steady_fps = (last[1] - first[1]) / dt
    return {
        "updates_logged": row_count,
        "final_update": rows[-1][0],
        "steady_fps": steady_fps,
    }


def render_table(results: list[dict]) -> str:
    rs_list = sorted({r["rollout_steps"] for r in results})
    n_list = sorted({r["num_envs"] for r in results})
    fps_grid = {(r["num_envs"], r["rollout_steps"]): r["steady_fps"] for r in results}

    lines = []
    header = f"{'num_envs':>10} | " + " | ".join(f"rs={rs:>5}" for rs in rs_list)
    lines.append(header)
    lines.append("-" * len(header))
    for n in n_list:
        cells = [
            f"{fps_grid[(n, rs)]:>7.0f}" if (n, rs) in fps_grid else f"{'-':>7}"
            for rs in rs_list
        ]
        lines.append(f"{n:>10} | " + " | ".join(cells))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", nargs="+", type=int,
                   default=[8, 16, 24, 32, 48, 64])
    p.add_argument("--rollout-steps", nargs="+", type=int,
                   default=[128, 256, 512])
    p.add_argument("--updates-target", type=int, default=15,
                   help="total_timesteps per combo sized to produce ~this many updates")
    p.add_argument("--opponent", default="random",
                   help="default 'random' to isolate rollout throughput from model inference")
    p.add_argument("--deck-paths", nargs="+",
                   default=["assets/decks/blue_eyes.ydk", "assets/decks/dark_magician.ydk"])
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=DEVICE_CHOICES,
                   help="Forward to cli.train --device (default: cpu)")
    p.add_argument("--vec-env-type", default="subproc", choices=VEC_ENV_TYPES,
                   help="Forward to cli.train --vec-env-type (default: subproc)")
    args = p.parse_args()

    grid_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / grid_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"opponent={args.opponent}, decks={args.deck_paths}, "
          f"device={args.device}, vec_env_type={args.vec_env_type}")
    print(f"grid: num_envs={args.num_envs} × rollout_steps={args.rollout_steps}")
    print(f"results -> {out_dir}\n", flush=True)

    results: list[dict] = []
    for num_envs in args.num_envs:
        for rs in args.rollout_steps:
            total = args.updates_target * num_envs * rs
            log_path = out_dir / f"n{num_envs}_rs{rs}.log"
            base_dir = out_dir / f"run_n{num_envs}_rs{rs}"
            cmd = [
                sys.executable, "-m", "cli.train",
                "--num-envs", str(num_envs),
                "--rollout-steps", str(rs),
                "--total-timesteps", str(total),
                "--opponent", args.opponent,
                "--deck-paths", *args.deck_paths,
                "--rnn-type", "none",
                "--device", args.device,
                "--vec-env-type", args.vec_env_type,
                "--log-interval", str(args.log_interval),
                "--seed", str(args.seed),
                "--base-dir", str(base_dir),
            ]
            print(f"[{datetime.now():%H:%M:%S}] n={num_envs:>3d} rs={rs:>4d} "
                  f"total={total:>8d} ... ", flush=True, end="")
            t0 = time.time()
            try:
                with log_path.open("w") as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   check=True, cwd=ROOT)
                wall = time.time() - t0
                stats = parse_log(log_path)
                if stats is None:
                    print(f"done in {wall:>4.0f}s but parse failed (see {log_path.name})")
                    continue
                fps = stats["steady_fps"]
                results.append({
                    "num_envs": num_envs,
                    "rollout_steps": rs,
                    "total_timesteps": total,
                    "wall_seconds": wall,
                    **stats,
                })
                print(f"done in {wall:>4.0f}s, FPS={fps:>5.0f}")
            except subprocess.CalledProcessError as e:
                print(f"FAILED rc={e.returncode} (see {log_path.name})")

    if not results:
        print("\nno successful runs", file=sys.stderr)
        return 1

    print("\nSteady-state FPS")
    print(render_table(results))

    best = max(results, key=lambda r: r["steady_fps"])
    batch = best["num_envs"] * best["rollout_steps"]
    print(f"\nBEST: num_envs={best['num_envs']} rollout_steps={best['rollout_steps']} "
          f"-> {best['steady_fps']:.0f} FPS  (batch={batch})")

    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), "results": results, "best": best}, indent=2,
    ))
    print(f"\nSummary written to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
