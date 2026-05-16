"""Auto-derived feature signature for a checkpoint.

Captured from the checkpoint's stored ``TrainingConfig``. ``compare`` groups
entries by these fields. The returned dict has stable key ordering so
identical configs produce byte-identical JSON.
"""

from __future__ import annotations

from typing import Any

from yugioh_rl.config import TrainingConfig, normalize_legacy_config


GROUPING_FIELDS: tuple[str, ...] = (
    "rnn_type",
    "rnn_hidden_dim",
    "rnn_num_layers",
    "bptt_chunk_len",
    "reward_shaping",
    "shaping_lp_weight",
    "shaping_card_weight",
    "agent_player",
    "deck_paths",
    "total_timesteps",
    "seed",
    "learning_rate",
    "entropy_coef",
    "clip_range",
    "card_embeddings",
    "training_opponent",
)


def extract_features(cfg: TrainingConfig) -> dict[str, Any]:
    """Build the ``features`` dict stored on each entry.

    Forward-compat: missing fields on a legacy pickled config are back-filled
    via ``normalize_legacy_config`` before extraction.
    """
    cfg = normalize_legacy_config(cfg)
    out: dict[str, Any] = {}
    for name in GROUPING_FIELDS:
        if name == "card_embeddings":
            out[name] = "semantic" if cfg.card_embeddings else "symbolic"
        elif name == "training_opponent":
            out[name] = cfg.opponent
        else:
            out[name] = getattr(cfg, name)

    return dict(sorted(out.items()))
