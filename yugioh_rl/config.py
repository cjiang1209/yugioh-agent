"""Training hyperparameter configuration."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, get_args

VecEnvType = Literal["subproc", "sync_actor_learner", "async_actor_learner"]
VEC_ENV_TYPES: tuple[str, ...] = get_args(VecEnvType)


@dataclass
class TrainingConfig:
    """All hyperparameters for PPO training."""

    # Environment
    num_envs: int = 8
    deck_paths: list[str] = field(default_factory=lambda: ["assets/decks/blue_eyes.ydk"])
    opponent: str = "greedy"  # "random", "greedy", or "model:path/to/checkpoint.pt"
    self_play: bool = False
    self_play_pool_size: int = 10
    self_play_temperature: float = 1.0
    self_play_sampling: Literal["uniform", "pfsp"] = "uniform"
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
    card_embeddings: str = ""  # path to pre-computed text embeddings (empty = disabled)
    text_embed_dim: int = 64  # frozen text embedding projection dim (only when enabled)
    learned_embed_dim: int = 8  # trainable per-card embedding dim (Mode B only)
    desc_n_embed_dim: int = 8  # sysstring desc embedding dim (per-action effect string id)

    # Recurrent policy (default "none" = feed-forward, byte-identical to pre-RNN)
    rnn_type: Literal["none", "lstm", "gru"] = "none"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    bptt_chunk_len: int = 16  # must divide rollout_steps when rnn_type != "none"

    # Infrastructure
    init_checkpoint: str = ""  # Path to .pt checkpoint to init weights from (new run)
    resume_checkpoint: str = ""  # Path to checkpoint to resume training from
    init_optimizer: bool = (
        False  # When using --init-checkpoint, also load optimizer state (not just weights)
    )
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 100
    eval_opponents: list[str] = field(default_factory=lambda: ["greedy", "random"])
    save_interval: int = 100
    save_dir: str = (
        "checkpoints"  # Exact run directory (CLI builds this from --base-dir + timestamp)
    )
    device: str = "auto"

    # "sync_actor_learner": workers hold a local policy and submit full
    # rollouts, avoiding the per-step IPC round-trip of "subproc".
    vec_env_type: VecEnvType = "subproc"

    # Async actor-learner: workers run continuously without sync barriers.
    # Rollouts with version lag > max_version_lag are discarded.
    max_version_lag: int = 5
    vtrace_rho_bar: float = 1.0  # V-trace IS truncation (async advantage estimation)
    vtrace_c_bar: float = 1.0  # V-trace trace-cutting coefficient

    @property
    def is_recurrent(self) -> bool:
        """Whether the network has an RNN module — drives TBPTT vs feed-forward."""
        return self.rnn_type != "none"


def normalize_legacy_config(cfg: TrainingConfig) -> TrainingConfig:
    """Back-fill new fields missing on a pickled TrainingConfig with their
    dataclass defaults. Mutates in place and returns the same instance.

    Single source of truth for forward-compatibility when loading older
    checkpoints. Must be called at every site that unpickles
    ``checkpoint["config"]`` before any code that uses ``vars()`` /
    ``cfg.__dict__`` (e.g. schema-drift detection, ``asdict`` consumers
    that round-trip through dicts).

    Membership is tested against ``cfg.__dict__`` rather than ``hasattr``
    because dataclass defaults live as class attributes — ``hasattr``
    would return True via class fallback even when the field is absent
    from the pickled instance state.
    """
    for f in fields(TrainingConfig):
        if f.name in cfg.__dict__:
            continue
        if f.default is not MISSING:
            setattr(cfg, f.name, f.default)
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            setattr(cfg, f.name, f.default_factory())
    return cfg
