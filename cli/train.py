"""PPO training entry point for Yu-Gi-Oh! RL agent."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Yu-Gi-Oh! RL agent with PPO")

    env = parser.add_argument_group("environment")
    env.add_argument("--num-envs", type=int, default=8,
                     help="Number of parallel environments (default: 8)")
    env.add_argument("--deck-path", type=str, default="assets/decks/starter.ydk",
                     help="Path to .ydk deck file used for both players (default: assets/decks/starter.ydk)")
    env.add_argument("--opponent", type=str, default="greedy", choices=["random", "greedy", "model"],
                     help="Opponent strategy: 'random' picks uniformly, 'greedy' picks highest-ATK, "
                          "'model' uses a trained checkpoint (default: greedy)")
    env.add_argument("--opponent-checkpoint", type=str, default="",
                     help="Path to .pt checkpoint for 'model' opponent (required when --opponent=model)")
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
    infra.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Compute device: 'auto' picks cuda if available (default: auto)")

    return parser.parse_args()


def _fatal(msg: str) -> None:
    """Print an error message to stderr and exit."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _was_provided(name: str) -> bool:
    """Check whether a CLI flag was explicitly passed by the user.

    Handles both ``--flag value`` and ``--flag=value`` forms.
    """
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI argument constraints that argparse cannot express.

    Logs warnings for arguments that have no effect given the other
    arguments provided.  Fatal constraint violations call ``_fatal``
    (and never return).
    """
    logger = logging.getLogger(__name__)

    # --resume is mutually exclusive with --init-checkpoint and --resume-optimizer
    if args.resume:
        if args.init_checkpoint:
            _fatal("--resume and --init-checkpoint are mutually exclusive")
        if args.resume_optimizer:
            _fatal("--resume-optimizer is for use with --init-checkpoint, not --resume")
        if not Path(args.resume).exists():
            _fatal(f"resume checkpoint not found: {args.resume}")
        if _was_provided("--base-dir"):
            logger.warning(
                "--base-dir has no effect with --resume "
                "(save directory is inferred from checkpoint path)"
            )

    # --resume-optimizer requires --init-checkpoint (when not using --resume)
    if args.resume_optimizer and not args.init_checkpoint:
        _fatal("--resume-optimizer requires --init-checkpoint")

    # --no-reward-shaping voids shaping weight arguments
    if args.no_reward_shaping:
        if _was_provided("--shaping-lp-weight"):
            logger.warning(
                "--shaping-lp-weight has no effect with --no-reward-shaping"
            )
        if _was_provided("--shaping-card-weight"):
            logger.warning(
                "--shaping-card-weight has no effect with --no-reward-shaping"
            )

    # --opponent-checkpoint only applies to --opponent=model
    if args.opponent_checkpoint and args.opponent != "model":
        logger.warning(
            "--opponent-checkpoint has no effect without --opponent=model"
        )

    # --eval-opponents validation
    for spec in args.eval_opponents:
        if spec.startswith("model:"):
            path = spec[len("model:"):]
            if not path:
                _fatal("--eval-opponents model: entries must include a checkpoint path")
            if not Path(path).exists():
                _fatal(f"eval opponent checkpoint not found: {path}")
        elif spec not in ("greedy", "random"):
            _fatal(f"unknown eval opponent: {spec}")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    validate_args(args)

    # Derive save_dir: resume continues in the same run directory
    if args.resume:
        save_dir = str(Path(args.resume).resolve().parent)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_seed{args.seed}"
        save_dir = str(Path(args.base_dir) / run_name)

    # Lazy import so --help is fast even without torch installed
    import torch
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        num_envs=args.num_envs,
        deck_path=args.deck_path,
        opponent_type=args.opponent,
        opponent_checkpoint=args.opponent_checkpoint,
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
    )

    # Create run directory and write config snapshot
    run_dir = Path(config.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info("Run directory: %s", run_dir)

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
