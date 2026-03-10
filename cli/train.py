"""PPO training entry point for Yu-Gi-Oh! RL agent."""

from __future__ import annotations

import argparse
import logging
import random
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Yu-Gi-Oh! RL agent with PPO")

    env = parser.add_argument_group("environment")
    env.add_argument("--num-envs", type=int, default=8,
                     help="Number of parallel environments (default: 8)")
    env.add_argument("--deck-path", type=str, default="assets/decks/starter.ydk",
                     help="Path to .ydk deck file used for both players (default: assets/decks/starter.ydk)")
    env.add_argument("--opponent", type=str, default="greedy", choices=["random", "greedy"],
                     help="Opponent strategy: 'random' picks uniformly, 'greedy' picks highest-ATK (default: greedy)")
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

    infra = parser.add_argument_group("infrastructure")
    infra.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    infra.add_argument("--log-interval", type=int, default=10,
                       help="Log metrics every N updates (default: 10)")
    infra.add_argument("--eval-interval", type=int, default=50,
                       help="Evaluate vs random/greedy every N updates (default: 50)")
    infra.add_argument("--eval-episodes", type=int, default=100,
                       help="Episodes per evaluation run (default: 100)")
    infra.add_argument("--save-interval", type=int, default=100,
                       help="Save checkpoint every N updates (default: 100)")
    infra.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints and TensorBoard logs (default: checkpoints)")
    infra.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Compute device: 'auto' picks cuda if available (default: auto)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy import so --help is fast even without torch installed
    import torch
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.ppo import PPOTrainer

    config = TrainingConfig(
        num_envs=args.num_envs,
        deck_path=args.deck_path,
        opponent_type=args.opponent,
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
        seed=args.seed,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        device=args.device,
    )

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
