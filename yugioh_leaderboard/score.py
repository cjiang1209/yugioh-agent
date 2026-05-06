"""Run the panel against a checkpoint and produce/update an Entry.

Does NOT regenerate ``index.md`` — that's the caller's responsibility
(the CLI calls ``index.write_index_file`` after ``score_checkpoint`` returns).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from yugioh_leaderboard.entry import (
    Entry,
    PanelMatchResult,
    compute_checkpoint_hash,
    deck_summary,
    entry_id_for,
    now_iso,
    stable_seed,
)
from yugioh_leaderboard.features import extract_features
from yugioh_leaderboard.panel import PanelConfig


def score_checkpoint(
    checkpoint_path: Path | str,
    panel: PanelConfig,
    *,
    deck_paths_override: Optional[list[str]] = None,
    episodes_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    tags: Optional[list[str]] = None,
    existing_entry: Optional[Entry] = None,
    precomputed_hash: Optional[str] = None,
) -> Entry:
    """Score ``checkpoint_path`` against ``panel`` and return an Entry.

    When ``existing_entry`` is provided, its ``pairwise_results`` are
    preserved (only ``panel_results`` and timestamps update). The CLI may
    pass ``precomputed_hash`` so we don't re-stream a multi-hundred-MB file
    that was already hashed for the dedup short-circuit.
    """
    import torch

    from cli.utils import resolve_device
    from yugioh_rl.config import TrainingConfig, normalize_legacy_config
    from yugioh_rl.env_wrapper import parse_deck_pool
    from yugioh_rl.eval import evaluate

    checkpoint_path = Path(checkpoint_path)
    eid = entry_id_for(checkpoint_path)
    chash = precomputed_hash if precomputed_hash is not None else compute_checkpoint_hash(checkpoint_path)
    device = resolve_device(panel.match.device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: TrainingConfig = normalize_legacy_config(ckpt["config"])
    del ckpt
    features = extract_features(cfg)

    deck_paths = deck_paths_override or list(features["deck_paths"])
    deck_pool = parse_deck_pool(deck_paths)

    episodes = episodes_override if episodes_override is not None else panel.match.episodes
    base_seed = seed_override if seed_override is not None else stable_seed(eid, "panel")

    panel_specs = [p.spec for p in panel.panel]
    panel_labels = [p.label for p in panel.panel]
    raw_results = evaluate(
        agent_spec=f"model:{checkpoint_path}",
        deck_pool=deck_pool,
        opponent_specs=panel_specs,
        num_episodes=episodes,
        seed=base_seed,
        agent_player=panel.match.agent_player,
        opponent_device=device,
        agent_device=device,
    )

    deck_stems = [Path(p).stem for p in deck_paths]
    panel_results: list[PanelMatchResult] = []
    for label, r in zip(panel_labels, raw_results):
        per_deck = {
            deck_stems[deck_idx]: deck_summary(int(sum(wl)), len(wl))
            for deck_idx, wl in r.per_deck_wins.items()
        }
        panel_results.append(
            PanelMatchResult(
                opponent_label=label,
                episodes=r.episodes,
                wins=r.wins,
                win_rate=r.win_rate,
                per_deck=per_deck,
                seed=base_seed,
                evaluated_at=now_iso(),
            )
        )

    if existing_entry is not None:
        return replace(
            existing_entry,
            checkpoint_hash=chash,
            added_at=now_iso(),
            panel_version=panel.panel_version,
            features=features,
            tags=list(tags) if tags is not None else existing_entry.tags,
            panel_results=panel_results,
        )

    return Entry(
        schema_version=1,
        entry_id=eid,
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=chash,
        added_at=now_iso(),
        panel_version=panel.panel_version,
        features=features,
        tags=list(tags) if tags else [],
        panel_results=panel_results,
        pairwise_results=[],
    )
