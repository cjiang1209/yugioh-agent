"""PPO training entry point for Yu-Gi-Oh! RL agent."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from cli.utils import (
    DEVICE_CHOICES,
    fatal,
    validate_deck_paths,
    validate_opponent_spec,
    was_provided,
)
from yugioh_rl.config import VEC_ENV_TYPES, TrainingConfig, normalize_legacy_config


# Flags whose values may override the checkpoint's stored config on --resume.
# Map CLI flag → TrainingConfig field name.
_RESUME_OVERRIDE_ALLOWLIST: dict[str, str] = {
    "--total-timesteps": "total_timesteps",
    "--learning-rate": "learning_rate",
    "--device": "device",
    "--log-interval": "log_interval",
    "--eval-interval": "eval_interval",
    "--eval-episodes": "eval_episodes",
    "--eval-opponents": "eval_opponents",
    "--save-interval": "save_interval",
    "--opponent": "opponent",
}

# Flags that are legal alongside --resume but do not correspond to a
# TrainingConfig override. --base-dir is tolerated with a warning in
# validate_cli_args; the others are session-scoped meta controls.
_RESUME_META_FLAGS: frozenset[str] = frozenset({
    "--resume", "--init-checkpoint", "--resume-optimizer", "--base-dir",
})

# TrainingConfig fields that are session-scoped; drop from ckpt_config on merge.
_META_FIELDS: frozenset[str] = frozenset(
    {"resume_checkpoint", "init_checkpoint", "resume_optimizer", "save_dir"}
)


def _provided_flags() -> list[str]:
    """Return the flag names the user explicitly passed on the command line.

    Scans ``sys.argv`` for ``--foo`` / ``--foo=bar`` tokens. Relies on
    argparse having already accepted the argv (so every ``--foo`` we see is
    a known, valid flag) — this function must only be called after
    ``parse_args()``.
    """
    flags = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            flags.append(arg.split("=", 1)[0])
    return flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Yu-Gi-Oh! RL agent with PPO")

    env = parser.add_argument_group("environment")
    env.add_argument("--num-envs", type=int, default=8,
                     help="Number of parallel environments (default: 8)")
    env.add_argument("--deck-paths", nargs="+", default=["assets/decks/starter.ydk"],
                     help="One or more .ydk deck files; agent and opponent sample from this pool "
                          "each episode (default: assets/decks/starter.ydk)")
    env.add_argument("--opponent", type=str, default="greedy",
                     help="Opponent spec: 'random', 'greedy', or 'model:path/to/checkpoint.pt' "
                          "(default: greedy)")
    env.add_argument("--no-reward-shaping", action="store_true",
                     help="Disable LP/card-advantage reward shaping (use sparse win/loss only)")
    env.add_argument("--shaping-lp-weight", type=float, default=0.01,
                     help="Weight for LP-delta shaping term (default: 0.01)")
    env.add_argument("--shaping-card-weight", type=float, default=0.005,
                     help="Weight for card-advantage shaping term (default: 0.005)")
    env.add_argument("--agent-player", type=str, default="random",
                     choices=["first", "second", "random"],
                     help="Agent turn order: 'first' (player 0), 'second' (player 1), "
                          "or 'random' (coin flip per episode, default: random)")

    ppo = parser.add_argument_group("PPO algorithm")
    ppo.add_argument("--total-timesteps", type=int, default=1_000_000,
                     help="Total env steps across all envs (default: 1000000)")
    ppo.add_argument("--rollout-steps", type=int, default=256,
                     help="Steps per env per rollout collection (default: 256)")
    ppo.add_argument("--num-epochs", type=int, default=4,
                     help="PPO optimization epochs per rollout (default: 4)")
    ppo.add_argument("--minibatch-size", type=int, default=256,
                     help="Minibatch size for PPO updates (default: 256)")
    ppo.add_argument("--learning-rate", type=float, default=3e-4,
                     help="Adam learning rate (default: 3e-4)")
    ppo.add_argument("--gamma", type=float, default=0.99,
                     help="Discount factor (default: 0.99)")
    ppo.add_argument("--gae-lambda", type=float, default=0.95,
                     help="GAE lambda for advantage estimation (default: 0.95)")
    ppo.add_argument("--clip-range", type=float, default=0.2,
                     help="PPO clipping epsilon (default: 0.2)")
    ppo.add_argument("--value-loss-coef", type=float, default=0.5,
                     help="Value loss coefficient (default: 0.5)")
    ppo.add_argument("--entropy-coef", type=float, default=0.01,
                     help="Entropy bonus coefficient (default: 0.01)")
    ppo.add_argument("--max-grad-norm", type=float, default=0.5,
                     help="Max gradient norm for clipping (default: 0.5)")

    net = parser.add_argument_group("network architecture")
    net.add_argument("--card-embed-dim", type=int, default=64,
                     help="Card encoder output dimension (default: 64)")
    net.add_argument("--global-embed-dim", type=int, default=64,
                     help="Global state encoder dimension (default: 64)")
    net.add_argument("--board-hidden-dim", type=int, default=256,
                     help="Board representation hidden dimension (default: 256)")
    net.add_argument("--action-embed-dim", type=int, default=64,
                     help="Action encoder output dimension (default: 64)")
    net.add_argument("--card-embeddings", type=str, default="",
                     help="Path to pre-computed card text embeddings .pt file (enables text-aware mode)")
    net.add_argument("--text-embed-dim", type=int, default=64,
                     help="Projected text embedding dimension (default: 64)")
    net.add_argument("--learned-embed-dim", type=int, default=8,
                     help="Collision-free learned embedding dimension in text mode (default: 8)")
    net.add_argument("--rnn-type", type=str, default="none",
                     choices=["none", "lstm", "gru"],
                     help="Recurrent layer between board MLP and the policy/value heads "
                          "(default: none = feed-forward)")
    net.add_argument("--rnn-hidden-dim", type=int, default=256,
                     help="Hidden size of the recurrent layer (default: 256)")
    net.add_argument("--rnn-num-layers", type=int, default=1,
                     help="Number of stacked recurrent layers (default: 1)")
    net.add_argument("--bptt-chunk-len", type=int, default=16,
                     help="Truncated-BPTT chunk length; must divide rollout_steps when an RNN "
                          "is enabled (default: 16). Ignored when --rnn-type=none.")

    infra = parser.add_argument_group("infrastructure")
    infra.add_argument("--resume", type=str, default="",
                       help="Path to .pt checkpoint to resume training from (continues in same run directory)")
    infra.add_argument("--init-checkpoint", type=str, default="",
                       help="Path to .pt checkpoint to initialize model weights from (starts a new run)")
    infra.add_argument("--resume-optimizer", action="store_true",
                       help="Also load optimizer state from checkpoint (use with --init-checkpoint)")
    infra.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    infra.add_argument("--log-interval", type=int, default=10,
                       help="Log metrics every N updates (default: 10)")
    infra.add_argument("--eval-interval", type=int, default=50,
                       help="Evaluate every N updates (default: 50)")
    infra.add_argument("--eval-opponents", nargs="+", default=["greedy", "random"],
                       help="Opponent specs for evaluation, e.g. 'greedy', 'random', "
                            "'model:checkpoints/latest.pt' (default: greedy random)")
    infra.add_argument("--eval-episodes", type=int, default=100,
                       help="Episodes per evaluation run (default: 100)")
    infra.add_argument("--save-interval", type=int, default=100,
                       help="Save checkpoint every N updates (default: 100)")
    infra.add_argument("--base-dir", type=str, default="checkpoints",
                       help="Base directory for runs; each run creates a timestamped "
                            "subdirectory (default: checkpoints)")
    infra.add_argument("--device", type=str, default="auto", choices=DEVICE_CHOICES,
                       help="Compute device: 'auto' picks cuda if available, else mps, else cpu (default: auto).")
    infra.add_argument("--vec-env-type", type=str, default="subproc",
                       choices=VEC_ENV_TYPES,
                       help="Vec-env transport: 'subproc' (synchronous IPC) or "
                            "'sync_actor_learner' (workers hold a local policy and submit "
                            "full rollouts; eliminates per-step round-trip).")

    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI args that do not depend on the resumed-config merge.

    Handles: --resume / --init-checkpoint mutual exclusion and path existence,
    --resume-optimizer legality, and "ignored flag" warnings.  Field-value
    checks (deck existence, opponent specs) run later against the effective
    TrainingConfig so that on --resume they see checkpoint values, not the
    discarded CLI args.
    """
    logger = logging.getLogger(__name__)

    # --resume is mutually exclusive with --init-checkpoint and --resume-optimizer
    if args.resume:
        if args.init_checkpoint:
            fatal("--resume and --init-checkpoint are mutually exclusive")
        if args.resume_optimizer:
            fatal("--resume-optimizer is for use with --init-checkpoint, not --resume")
        if not Path(args.resume).exists():
            fatal(f"resume checkpoint not found: {args.resume}")
        if was_provided("--base-dir"):
            logger.warning(
                "--base-dir has no effect with --resume "
                "(save directory is inferred from checkpoint path)"
            )

    # --resume-optimizer requires --init-checkpoint (when not using --resume)
    if args.resume_optimizer and not args.init_checkpoint:
        fatal("--resume-optimizer requires --init-checkpoint")

    # --no-reward-shaping voids shaping weight arguments
    if args.no_reward_shaping:
        if was_provided("--shaping-lp-weight"):
            logger.warning(
                "--shaping-lp-weight has no effect with --no-reward-shaping"
            )
        if was_provided("--shaping-card-weight"):
            logger.warning(
                "--shaping-card-weight has no effect with --no-reward-shaping"
            )


def validate_effective_config(config: "TrainingConfig") -> None:  # noqa: F821
    """Validate per-field values on the final merged TrainingConfig."""
    validate_deck_paths(config.deck_paths)
    validate_opponent_spec(config.opponent, "--opponent")
    for spec in config.eval_opponents:
        validate_opponent_spec(spec, "--eval-opponents")

    if config.bptt_chunk_len < 1:
        fatal(f"--bptt-chunk-len must be >= 1 (got {config.bptt_chunk_len})")

    if not config.is_recurrent:
        return

    # TBPTT invariants — derived from the envs-as-unit batching algorithm
    # in RolloutBuffer.get_recurrent_batches.  Each one names the conflicting
    # fields so the user can see exactly what to drop.
    if config.rollout_steps % config.bptt_chunk_len != 0:
        fatal(
            f"with --rnn-type={config.rnn_type}, --rollout-steps "
            f"({config.rollout_steps}) must be divisible by --bptt-chunk-len "
            f"({config.bptt_chunk_len}) so the rollout splits into whole chunks"
        )
    if config.minibatch_size < config.rollout_steps:
        fatal(
            f"with --rnn-type={config.rnn_type}, --minibatch-size "
            f"({config.minibatch_size}) must be >= --rollout-steps "
            f"({config.rollout_steps}); minibatches are env-grouped, not "
            f"flattened. Try --minibatch-size {config.rollout_steps}."
        )
    if config.minibatch_size % config.rollout_steps != 0:
        fatal(
            f"with --rnn-type={config.rnn_type}, --minibatch-size "
            f"({config.minibatch_size}) must be divisible by --rollout-steps "
            f"({config.rollout_steps}) so envs_per_minibatch is an integer."
        )
    if config.num_envs * config.rollout_steps < config.minibatch_size:
        fatal(
            f"--minibatch-size ({config.minibatch_size}) > "
            f"--num-envs * --rollout-steps "
            f"({config.num_envs * config.rollout_steps}); the rollout has "
            f"fewer total samples than one minibatch."
        )
    envs_per_minibatch = config.minibatch_size // config.rollout_steps
    if config.num_envs % envs_per_minibatch != 0:
        fatal(
            f"--num-envs ({config.num_envs}) must be divisible by "
            f"envs_per_minibatch ({envs_per_minibatch}, derived from "
            f"--minibatch-size / --rollout-steps) so every epoch yields "
            f"equal-sized minibatches."
        )


def _build_fresh_config(args: argparse.Namespace, save_dir: str) -> TrainingConfig:
    """Build TrainingConfig directly from CLI args (fresh / --init-checkpoint)."""
    return TrainingConfig(
        num_envs=args.num_envs,
        deck_paths=args.deck_paths,
        opponent=args.opponent,
        agent_player=args.agent_player,
        reward_shaping=not args.no_reward_shaping,
        shaping_lp_weight=args.shaping_lp_weight,
        shaping_card_weight=args.shaping_card_weight,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        card_embed_dim=args.card_embed_dim,
        global_embed_dim=args.global_embed_dim,
        board_hidden_dim=args.board_hidden_dim,
        action_embed_dim=args.action_embed_dim,
        card_embeddings_path=args.card_embeddings,
        text_embed_dim=args.text_embed_dim,
        learned_embed_dim=args.learned_embed_dim,
        rnn_type=args.rnn_type,
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_num_layers=args.rnn_num_layers,
        bptt_chunk_len=args.bptt_chunk_len,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume,
        resume_optimizer=args.resume_optimizer,
        seed=args.seed,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_opponents=args.eval_opponents,
        save_interval=args.save_interval,
        save_dir=save_dir,
        device=args.device,
        vec_env_type=args.vec_env_type,
    )


def _build_resume_config(args: argparse.Namespace, save_dir: str) -> TrainingConfig:
    """Load ``ckpt["config"]``, merge CLI allowlist overrides, return a
    TrainingConfig ready for PPOTrainer.

    Hard-errors if the user passed any non-allowlisted override flag or if
    the checkpoint config's field set has drifted from the current
    ``TrainingConfig`` schema.
    """
    provided = set(_provided_flags())

    # Reject non-allowlist CLI overrides before any expensive I/O so the
    # user sees a clear "this flag isn't overridable" error first.
    disallowed = sorted(
        provided - set(_RESUME_OVERRIDE_ALLOWLIST) - _RESUME_META_FLAGS
    )
    if disallowed:
        allowlist_str = ", ".join(sorted(_RESUME_OVERRIDE_ALLOWLIST.keys()))
        fatal(
            "these flags cannot be overridden on --resume: "
            + ", ".join(disallowed)
            + f"; allowed: {allowlist_str}"
        )

    import torch

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config")
    if ckpt_config is None:
        fatal(f"resume checkpoint has no stored config: {args.resume}")
    if not isinstance(ckpt_config, TrainingConfig):
        fatal(
            f"resume checkpoint config is not a TrainingConfig: "
            f"{type(ckpt_config).__name__}"
        )

    # Back-fill any fields the pickled config is missing (added since the
    # checkpoint was saved). Lets the schema-drift check below silently
    # accept additive changes while still catching renames/removals.
    normalize_legacy_config(ckpt_config)

    # Schema-drift detection: compare pickled instance attrs (what was
    # actually stored) against the current TrainingConfig field set.
    # Using vars() rather than asdict().keys() so extras survive the check —
    # asdict walks the current class's fields() and would silently drop them.
    ckpt_attrs = set(vars(ckpt_config).keys())
    current_fields = {f.name for f in fields(TrainingConfig)}
    missing = current_fields - ckpt_attrs
    extra = ckpt_attrs - current_fields
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unknown: {sorted(extra)}")
        fatal(
            "resume checkpoint config does not match current TrainingConfig "
            "schema (" + "; ".join(parts)
            + "). Use a matching codebase version."
        )

    # Start from the checkpoint's stored values; drop session-scoped fields
    # (they must come from the current CLI invocation, not the old run).
    merged = {k: v for k, v in asdict(ckpt_config).items() if k not in _META_FIELDS}

    # Apply allowlisted CLI overrides (only when the user explicitly passed
    # the flag — otherwise keep the checkpoint's value).  The dict's values
    # match argparse's default dest for each flag, so they're both the
    # TrainingConfig field name and the attribute on `args`.
    for flag, field_name in _RESUME_OVERRIDE_ALLOWLIST.items():
        if flag in provided:
            merged[field_name] = getattr(args, field_name)

    # Meta fields come from the CLI invocation.
    merged["resume_checkpoint"] = args.resume
    merged["init_checkpoint"] = ""
    merged["resume_optimizer"] = False
    merged["save_dir"] = save_dir

    return TrainingConfig(**merged)


def _write_config_snapshot(config: TrainingConfig) -> Path:
    """Write ``config_{timestamp}.json`` into ``config.save_dir`` and repoint
    the ``config.json`` symlink at it."""
    run_dir = Path(config.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"config_{timestamp}.json"
    snapshot_path = run_dir / snapshot_name
    with open(snapshot_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    latest = run_dir / "config.json"
    latest.unlink(missing_ok=True)
    latest.symlink_to(snapshot_name)
    return snapshot_path


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    validate_cli_args(args)

    # Derive save_dir: resume continues in the same run directory
    if args.resume:
        save_dir = str(Path(args.resume).resolve().parent)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_seed{args.seed}"
        save_dir = str(Path(args.base_dir) / run_name)

    # On fresh runs, validate effective config before importing torch so
    # bad --opponent / --deck-paths fail fast without paying torch's
    # import cost.  Resume requires torch.load to read ckpt["config"]
    # before we can validate — _build_resume_config imports torch itself.
    if args.resume:
        config = _build_resume_config(args, save_dir)
    else:
        config = _build_fresh_config(args, save_dir)
    validate_effective_config(config)

    import torch
    from yugioh_rl.ppo import PPOTrainer

    # Create run directory and write a timestamped config snapshot; update
    # `config.json` symlink to point at it.
    run_dir = Path(config.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_config_snapshot(config)
    logger.info("Run directory: %s", run_dir)
    logger.info("Config snapshot: %s", snapshot_path.name)
    if len(config.deck_paths) > 1:
        logger.info("Multi-deck training: %d decks — %s", len(config.deck_paths),
                     ", ".join(config.deck_paths))

    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
