"""Pure statistics + group-comparison logic. No I/O.

Variance unit is the seed (one win-rate per training seed). Bootstrap CIs
are over seeds, not over individual episodes — a 100-game match against
greedy is one correlated sample of a checkpoint's strength, not 100
independent ones.

The bootstrap RNG seed is baked in (constant ``42``) so identical input
data produces byte-identical CIs across runs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from yugioh_leaderboard.entry import Entry

_BOOTSTRAP_SEED = 42
_DEFAULT_ITERS = 10_000
_DEFAULT_ALPHA = 0.05


_HASH_MASK = (1 << 64) - 1


def _seed_for(values: np.ndarray) -> tuple[int, int]:
    """Per-call RNG seed: deterministic in input data but distinct across inputs.

    Re-seeding to a global constant on every call would make every CI in a
    single comparison report use the *identical* resample-index pattern,
    artificially correlating draws across opponents and groups. Hashing the
    input data preserves run-to-run determinism without that side effect.
    """
    return (_BOOTSTRAP_SEED, hash(values.tobytes()) & _HASH_MASK)


def _resample_quantiles(arr: np.ndarray, n_iters: int, alpha: float) -> tuple[float, float]:
    """Resample ``arr`` with replacement and return ``(lo, hi)`` percentile bounds."""
    rng = np.random.default_rng(_seed_for(arr))
    means = arr[rng.integers(0, arr.size, size=(n_iters, arr.size))].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci(
    values: list[float],
    *,
    n_iters: int = _DEFAULT_ITERS,
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Resample ``values`` with replacement; return (low, high) percentile CI.

    Edge cases:
      * Empty input → (0.0, 0.0).
      * Single value → (v, v).
      * All identical → (v, v).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    return _resample_quantiles(arr, n_iters, alpha)


def paired_bootstrap_delta(
    group_a: dict[int, float],
    group_b: dict[int, float],
    *,
    n_iters: int = _DEFAULT_ITERS,
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Paired bootstrap of (group_b - group_a) over seeds in both dicts.

    Returns ``(delta_mean, ci_low, ci_high)``. Raises ``ValueError`` when
    the seed intersection is empty.
    """
    shared = sorted(set(group_a) & set(group_b))
    if not shared:
        raise ValueError("no shared seeds between groups — cannot pair")
    deltas = np.array([group_b[s] - group_a[s] for s in shared], dtype=np.float64)
    delta_mean = float(deltas.mean())
    if deltas.size == 1:
        return delta_mean, delta_mean, delta_mean
    lo, hi = _resample_quantiles(deltas, n_iters, alpha)
    return delta_mean, lo, hi


@dataclass
class GroupStats:
    """Per-group, per-opponent: mean win rate + bootstrap CI across seeds."""

    n_seeds: int
    seeds: list[int]
    by_opponent: dict[str, tuple[float, float, float]]


@dataclass
class ComparisonResult:
    by_field: str
    groups: OrderedDict[str, GroupStats]
    paired_delta_by_group: dict[str, dict[str, tuple[float, float, float]]] = field(
        default_factory=dict
    )
    baseline_group: str = ""
    skip_reason: str | None = None


def _entry_winrates_by_opponent(entry: Entry) -> dict[str, float]:
    return {r.opponent_label: r.win_rate for r in entry.panel_results}


def value_repr(value: object) -> str:
    """Stable, lower-cased string form for grouping/filtering (handles lists, bools)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(value_repr(x) for x in value)
    return str(value)


def matches_filter(entry: Entry, filter_: dict[str, object] | None) -> bool:
    """Filter values are compared case-insensitively so CLI strings like
    ``true`` match Python bool ``True`` (which ``value_repr`` lower-cases) and
    ``LSTM`` matches a feature literal ``"lstm"``."""
    if not filter_:
        return True
    for k, v in filter_.items():
        if value_repr(entry.features.get(k)).lower() != value_repr(v).lower():
            return False
    return True


def compare_groups(
    entries: list[Entry],
    *,
    by_field: str,
    filter: dict[str, object] | None = None,
    opponents: list[str] | None = None,
) -> ComparisonResult:
    """Group entries by ``features[by_field]`` and compute per-group stats."""
    matched = [e for e in entries if matches_filter(e, filter)]

    by_group: dict[str, list[Entry]] = {}
    for e in matched:
        if by_field not in e.features:
            continue
        key = value_repr(e.features[by_field])
        by_group.setdefault(key, []).append(e)

    if len(by_group) < 2:
        if filter:
            reason = (
                f"filter left {len(by_group)} group(s) for `{by_field}`; need at least 2 to compare"
            )
        else:
            reason = f"only {len(by_group)} value of `{by_field}`"
        return ComparisonResult(
            by_field=by_field,
            groups=OrderedDict(),
            skip_reason=reason,
        )

    wr_by_entry: dict[int, dict[str, float]] = {
        id(e): _entry_winrates_by_opponent(e) for e in matched
    }

    if opponents is None:
        seen: list[str] = []
        for e in matched:
            for label in wr_by_entry[id(e)]:
                if label not in seen:
                    seen.append(label)
        opponents = seen

    groups: OrderedDict[str, GroupStats] = OrderedDict()
    for key in sorted(by_group):
        gs_entries = by_group[key]
        seeds = sorted(int(e.features.get("seed", 0)) for e in gs_entries)
        per_opp: dict[str, tuple[float, float, float]] = {}
        for opp in opponents:
            wrs = [wr_by_entry[id(e)][opp] for e in gs_entries if opp in wr_by_entry[id(e)]]
            if not wrs:
                continue
            mean = sum(wrs) / len(wrs)
            lo, hi = bootstrap_ci(wrs)
            per_opp[opp] = (mean, lo, hi)
        groups[key] = GroupStats(n_seeds=len(seeds), seeds=seeds, by_opponent=per_opp)

    baseline = next(iter(groups))

    def _seed_winrates(group_entries: list[Entry], opp: str) -> dict[int, float]:
        return {
            int(e.features.get("seed", 0)): wr_by_entry[id(e)][opp]
            for e in group_entries
            if opp in wr_by_entry[id(e)]
        }

    baseline_seed_winrates = {opp: _seed_winrates(by_group[baseline], opp) for opp in opponents}

    paired: dict[str, dict[str, tuple[float, float, float]]] = {}
    for key, gs_entries in by_group.items():
        if key == baseline:
            continue
        per_opp_delta: dict[str, tuple[float, float, float]] = {}
        for opp in opponents:
            target_seeds = _seed_winrates(gs_entries, opp)
            if not target_seeds or not baseline_seed_winrates[opp]:
                continue
            try:
                per_opp_delta[opp] = paired_bootstrap_delta(
                    baseline_seed_winrates[opp], target_seeds
                )
            except ValueError:
                continue
        paired[key] = per_opp_delta

    return ComparisonResult(
        by_field=by_field,
        groups=groups,
        paired_delta_by_group=paired,
        baseline_group=baseline,
    )


def _fmt_rate(v: float) -> str:
    return f"{v:.2f}"


def _fmt_signed(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


def format_comparison_table(result: ComparisonResult) -> str:
    """Render a ComparisonResult as a Markdown table.

    Bolds the paired delta when the CI excludes 0 (``**...**``), signaling
    a meaningful difference. Win-rate cells render as ``mean [lo, hi]`` so
    asymmetric bootstrap CIs near 0 or 1 stay visible.
    """
    if result.skip_reason:
        return f"no comparison: {result.skip_reason}\n"

    opponents = []
    for gs in result.groups.values():
        for opp in gs.by_opponent:
            if opp not in opponents:
                opponents.append(opp)

    headers = [
        result.by_field,
        "n",
        *(f"vs {opp}" for opp in opponents),
        f"Δ vs `{result.baseline_group}` (paired)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for key, gs in result.groups.items():
        row = [key, str(gs.n_seeds)]
        for opp in opponents:
            if opp in gs.by_opponent:
                mean, lo, hi = gs.by_opponent[opp]
                row.append(f"{_fmt_rate(mean)} [{_fmt_rate(lo)}, {_fmt_rate(hi)}]")
            else:
                row.append("—")
        if key == result.baseline_group:
            row.append("—")
        else:
            deltas = result.paired_delta_by_group.get(key, {})
            cells = []
            for opp in opponents:
                if opp not in deltas:
                    continue
                d, lo, hi = deltas[opp]
                meaningful = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
                txt = f"{_fmt_signed(d)} [{_fmt_signed(lo)}, {_fmt_signed(hi)}]"
                if meaningful:
                    txt = f"**{txt}**"
                cells.append(f"{opp}: {txt}")
            row.append("; ".join(cells) if cells else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
