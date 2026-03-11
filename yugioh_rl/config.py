"""Training hyperparameter configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """All hyperparameters for PPO training."""

    # Environment
    num_envs: int = 8
    deck_path: str = "assets/decks/starter.ydk"
    opponent_type: str = "greedy"
    opponent_checkpoint: str = ""  # Path to .pt checkpoint for "model" opponent
    agent_player: str = "random"  # "first" (player 0), "second" (player 1), or "random"
    reward_shaping: bool = True
    shaping_lp_weight: float = 0.01
    shaping_card_weight: float = 0.005

    # PPO
    total_timesteps: int = 1_000_000
    rollout_steps: int = 256
    num_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Network
    card_embed_dim: int = 64
    global_embed_dim: int = 64
    board_hidden_dim: int = 256
    action_embed_dim: int = 64

    # Infrastructure
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 100
    save_interval: int = 100
    save_dir: str = "checkpoints"
    device: str = "auto"
