"""PPO algorithm with rollout buffer for Yu-Gi-Oh! training."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
)
from yugioh_env.opponent import NetworkOpponent
from yugioh_rl.config import TrainingConfig
from yugioh_rl.env_wrapper import SubprocVecEnv, parse_deck_pool
from yugioh_rl.eval import evaluate_with_agent, log_results_to_tensorboard
from yugioh_rl.network import YuGiOhNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray, device: torch.device, *, long: bool = False) -> torch.Tensor:
    """``np.ndarray`` → ``torch.Tensor`` on ``device``, with optional cast to int64."""
    t = torch.from_numpy(arr).to(device)
    return t.long() if long else t


@dataclass
class MiniBatch:
    """A single minibatch of training data (feed-forward path)."""

    obs_cards: torch.Tensor  # (M, MAX_CARDS, CARD_FEATURES)
    obs_global: torch.Tensor  # (M, GLOBAL_FEATURES)
    obs_actions: torch.Tensor  # (M, MAX_ACTIONS, ACTION_FEATURES)
    action_mask: torch.Tensor  # (M, MAX_ACTIONS)
    obs_chain: torch.Tensor  # (M, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)
    obs_event: torch.Tensor  # (M, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    actions: torch.Tensor  # (M,)
    old_log_probs: torch.Tensor  # (M,)
    advantages: torch.Tensor  # (M,)
    returns: torch.Tensor  # (M,)


@dataclass
class RecurrentMiniBatch:
    """A single TBPTT minibatch: full T-step rollout slice for env_mb envs.

    The PPO update walks chunks of length L through the leading T axis,
    threading hx with a detach() between chunks.  Shapes are (T, env_mb, ...)
    not (M, ...) — the chunk loop reshapes per-chunk to (L*env_mb, ...) for
    the network forward.
    """

    obs_cards: torch.Tensor  # (T, env_mb, MAX_CARDS, CARD_FEATURES)
    obs_global: torch.Tensor  # (T, env_mb, GLOBAL_FEATURES)
    obs_actions: torch.Tensor  # (T, env_mb, MAX_ACTIONS, ACTION_FEATURES)
    action_mask: torch.Tensor  # (T, env_mb, MAX_ACTIONS)
    obs_chain: torch.Tensor  # (T, env_mb, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)
    obs_event: torch.Tensor  # (T, env_mb, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
    actions: torch.Tensor  # (T, env_mb)
    old_log_probs: torch.Tensor  # (T, env_mb)
    advantages: torch.Tensor  # (T, env_mb)
    returns: torch.Tensor  # (T, env_mb)
    dones: torch.Tensor  # (T, env_mb)
    hx_initial: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None


class RolloutBuffer:
    """Stores trajectory data for PPO rollouts."""

    def __init__(self, rollout_steps: int, num_envs: int) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self._ptr = 0

        T, N = rollout_steps, num_envs
        self.obs_cards = np.zeros((T, N, MAX_CARDS, CARD_FEATURES), dtype=np.uint8)
        self.obs_global = np.zeros((T, N, GLOBAL_FEATURES), dtype=np.uint8)
        self.obs_actions = np.zeros((T, N, MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
        self.obs_mask = np.zeros((T, N, MAX_ACTIONS), dtype=np.int8)
        self.obs_chain = np.zeros((T, N, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES), dtype=np.uint8)
        self.obs_event = np.zeros((T, N, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), dtype=np.uint8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

        # Single snapshot of the recurrent hidden state at t=0, set by the
        # trainer right before the rollout loop runs.  Stored but unused by
        # the current update loop; the TBPTT path will consume it.
        self.hx_initial: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None

    def reset(self) -> None:
        self._ptr = 0
        self.hx_initial = None

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Store one timestep of data."""
        t = self._ptr
        self.obs_cards[t] = obs["cards"]
        self.obs_global[t] = obs["global_state"]
        self.obs_actions[t] = obs["actions"]
        self.obs_mask[t] = obs["action_mask"]
        self.obs_chain[t] = obs["pending_chain"]
        self.obs_event[t] = obs["event_history"]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self._ptr += 1

    def ingest_rollouts(self, rollouts: list[dict]) -> dict[str, np.ndarray]:
        """Bulk-load N actor-learner rollouts into the (T, N, ...) buffer.

        Per-env assignment avoids the ~tens-of-MB intermediate ``np.stack``
        would allocate. Returns the per-env final obs dict so the caller can
        drive GAE bootstrap without re-deriving it from the rollout payloads.
        """
        T = self.rollout_steps
        assert len(rollouts) == self.num_envs, (
            f"expected {self.num_envs} rollouts, got {len(rollouts)}"
        )
        for i, r in enumerate(rollouts):
            self.obs_cards[:T, i] = r["obs_cards"]
            self.obs_global[:T, i] = r["obs_global"]
            self.obs_actions[:T, i] = r["obs_actions"]
            self.obs_mask[:T, i] = r["action_mask"]
            self.obs_chain[:T, i] = r["obs_chain"]
            self.obs_event[:T, i] = r["obs_event"]
            self.actions[:T, i] = r["actions"]
            self.log_probs[:T, i] = r["log_probs"]
            self.values[:T, i] = r["values"]
            self.rewards[:T, i] = r["rewards"]
            self.dones[:T, i] = r["dones"].astype(np.float32)
        return {
            "cards": np.stack([r["final_obs_cards"] for r in rollouts]),
            "global_state": np.stack([r["final_obs_global"] for r in rollouts]),
            "actions": np.stack([r["final_obs_actions"] for r in rollouts]),
            "action_mask": np.stack([r["final_action_mask"] for r in rollouts]),
            "pending_chain": np.stack([r["final_obs_chain"] for r in rollouts]),
            "event_history": np.stack([r["final_obs_event"] for r in rollouts]),
        }

    def compute_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE-lambda advantages and returns."""
        T = self.rollout_steps
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, minibatch_size: int, device: torch.device) -> Iterator[MiniBatch]:
        """Yield shuffled minibatches as tensors on the given device."""
        T, N = self.rollout_steps, self.num_envs
        total = T * N

        # Flatten time and env dims
        flat_cards = self.obs_cards.reshape(total, MAX_CARDS, CARD_FEATURES)
        flat_global = self.obs_global.reshape(total, GLOBAL_FEATURES)
        flat_actions_obs = self.obs_actions.reshape(total, MAX_ACTIONS, ACTION_FEATURES)
        flat_mask = self.obs_mask.reshape(total, MAX_ACTIONS)
        flat_chain = self.obs_chain.reshape(total, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES)
        flat_event = self.obs_event.reshape(total, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)

        # Normalize advantages
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]

            yield MiniBatch(
                obs_cards=_to_tensor(flat_cards[idx], device),
                obs_global=_to_tensor(flat_global[idx], device),
                obs_actions=_to_tensor(flat_actions_obs[idx], device),
                action_mask=_to_tensor(flat_mask[idx], device),
                obs_chain=_to_tensor(flat_chain[idx], device),
                obs_event=_to_tensor(flat_event[idx], device),
                actions=_to_tensor(flat_actions[idx], device, long=True),
                old_log_probs=_to_tensor(flat_log_probs[idx], device),
                advantages=_to_tensor(flat_advantages[idx], device),
                returns=_to_tensor(flat_returns[idx], device),
            )

    def get_recurrent_batches(
        self,
        minibatch_size: int,
        device: torch.device,
    ) -> Iterator[RecurrentMiniBatch]:
        """Yield env-grouped TBPTT minibatches.

        ``envs_per_minibatch = minibatch_size // rollout_steps`` envs per
        minibatch; each minibatch carries the full T-step rollout for those
        envs plus the rollout-start ``hx_initial`` slice.  Advantage
        normalization is over the full rollout so cross-minibatch scaling
        is preserved.

        ``validate_effective_config`` enforces that ``num_envs`` is divisible
        by ``envs_per_minibatch``, so every minibatch has the same env_mb.
        """
        T, N = self.rollout_steps, self.num_envs
        envs_per_minibatch = minibatch_size // T

        # Normalize over the full rollout (matches the feed-forward path's
        # global normalization rather than per-minibatch).
        flat_adv = self.advantages.reshape(-1)
        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std() + 1e-8
        norm_adv = (self.advantages - adv_mean) / adv_std

        env_indices = np.arange(N)
        np.random.shuffle(env_indices)

        for start in range(0, N, envs_per_minibatch):
            env_idx_np = env_indices[start : start + envs_per_minibatch]
            env_idx_t = torch.from_numpy(env_idx_np).long().to(device)

            yield RecurrentMiniBatch(
                obs_cards=_to_tensor(self.obs_cards[:, env_idx_np], device),
                obs_global=_to_tensor(self.obs_global[:, env_idx_np], device),
                obs_actions=_to_tensor(self.obs_actions[:, env_idx_np], device),
                action_mask=_to_tensor(self.obs_mask[:, env_idx_np], device),
                obs_chain=_to_tensor(self.obs_chain[:, env_idx_np], device),
                obs_event=_to_tensor(self.obs_event[:, env_idx_np], device),
                actions=_to_tensor(self.actions[:, env_idx_np], device, long=True),
                old_log_probs=_to_tensor(self.log_probs[:, env_idx_np], device),
                advantages=_to_tensor(norm_adv[:, env_idx_np], device),
                returns=_to_tensor(self.returns[:, env_idx_np], device),
                dones=_to_tensor(self.dones[:, env_idx_np], device),
                hx_initial=YuGiOhNet.slice_hx(self.hx_initial, env_idx_t),
            )


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO training loop for Yu-Gi-Oh! agent."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # PyTorch 2.11 MPS LSTM backward kernel asserts during the per-step
        # TBPTT replay (forward succeeds, backward trips a uint32 underflow in
        # MPSNDArrayDescriptor). GRU and the whole-sequence LSTM path are both
        # fine, so this is specific to the per-step LSTM backward our TBPTT
        # update uses. Fail fast at construction with a usable workaround
        # rather than crashing several minutes into a run with a cryptic
        # Metal driver assertion. See bugs/mps_lstm_per_step_backward/ for repros.
        if self.device.type == "mps" and config.rnn_type == "lstm":
            raise RuntimeError(
                "rnn_type='lstm' on device='mps' triggers a PyTorch MPS "
                "backward-kernel crash during PPO updates. Workarounds: "
                "use --rnn-type gru, or --device cpu. "
                "(See bugs/mps_lstm_per_step_backward/mps_lstm_minimal.py for the minimal repro.)"
            )

        logger.info("Using device: %s", self.device)

        # Network, optimizer, and resume state
        self._resume_update = 0
        self._resume_global_step = 0
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_wins: list[float] = []
        self._deck_wins: dict[int, list[float]] = {}

        if config.resume_checkpoint:
            self._resume_update, self._resume_global_step = self._load_resume_checkpoint()
        elif config.init_checkpoint:
            ckpt = torch.load(config.init_checkpoint, map_location=self.device, weights_only=False)
            self._validate_checkpoint_compat(config, ckpt)
            self.network = YuGiOhNet.from_state_dict(config, ckpt["model_state_dict"]).to(
                self.device
            )
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
            if config.init_optimizer and "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Override LR from CLI so users can change schedule across runs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = config.learning_rate
            logger.info("Initialized weights from checkpoint: %s", config.init_checkpoint)
        else:
            self.network = YuGiOhNet.from_config(config).to(self.device)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

        self._opponent_pool = None
        if config.self_play:
            from yugioh_rl.opponent_pool import OpponentPool

            def network_factory():
                return YuGiOhNet.from_config(config)

            if config.resume_checkpoint:
                self._opponent_pool = OpponentPool.from_resume(
                    pool_size=config.self_play_pool_size,
                    initial_opponent_spec=config.opponent,
                    network_factory=network_factory,
                    save_interval=config.save_interval,
                    checkpoint_dir=Path(config.save_dir),
                    temperature=config.self_play_temperature,
                    sampling=config.self_play_sampling,
                )
            else:
                self._opponent_pool = OpponentPool.create_trainer(
                    pool_size=config.self_play_pool_size,
                    initial_opponent_spec=config.opponent,
                    network_factory=network_factory,
                    temperature=config.self_play_temperature,
                    sampling=config.self_play_sampling,
                )

        # Rollout buffer
        self.buffer = RolloutBuffer(config.rollout_steps, config.num_envs)

        # Deck pool (pre-parsed once; passed to env workers)
        self._deck_pool = parse_deck_pool(config.deck_paths)

        # TensorBoard writer (optional)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            purge = self._resume_global_step if self._resume_global_step > 0 else None
            self._writer = SummaryWriter(
                log_dir=str(Path(config.save_dir) / "logs"),
                purge_step=purge,
            )
        except ImportError:
            logger.info("TensorBoard not available, skipping logging")

    def _load_resume_checkpoint(self) -> tuple[int, int]:
        """Load full training state from a checkpoint for resumption.

        Returns ``(update, global_step)`` so the training loop can continue
        from the correct point.
        """
        config = self.config
        ckpt = torch.load(config.resume_checkpoint, map_location=self.device, weights_only=False)
        self._validate_checkpoint_compat(config, ckpt)

        self.network = YuGiOhNet.from_state_dict(config, ckpt["model_state_dict"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in self.optimizer.param_groups:
                pg["lr"] = config.learning_rate

        # Restore episode tracking (with fallback for old checkpoints).
        # Note: env RNG state is NOT restored — SubprocVecEnv re-seeds from
        # config.seed on creation, so the episode sequence after resume will
        # diverge from a single uninterrupted run.  This is acceptable;
        # saving per-worker RNG state is impractical with the multi-process
        # architecture.
        self._episode_rewards = list(ckpt.get("episode_rewards", []))
        self._episode_lengths = list(ckpt.get("episode_lengths", []))
        self._episode_wins = list(ckpt.get("episode_wins", []))
        self._deck_wins = {int(k): list(v) for k, v in ckpt.get("deck_wins", {}).items()}

        update = ckpt.get("update", 0)
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Loaded checkpoint for resumption: %s (update=%d, global_step=%d)",
            config.resume_checkpoint,
            update,
            global_step,
        )
        return update, global_step

    @staticmethod
    def _validate_checkpoint_compat(config: TrainingConfig, ckpt: dict) -> None:
        """Validate architecture compatibility before loading weights."""
        ckpt_config = ckpt.get("config")
        if ckpt_config is None:
            logger.warning("Checkpoint has no saved config — skipping compatibility check")
            return

        # Architecture fields that always determine layer shapes — must
        # match exactly.  rnn_type defaults to "none" so a pre-RNN
        # checkpoint resumed by post-RNN code compares cleanly; any
        # mismatch means the user is hot-adding or hot-removing a
        # recurrent layer, which would silently corrupt trained weights.
        arch_fields = [
            "card_embed_dim",
            "global_embed_dim",
            "board_hidden_dim",
            "action_embed_dim",
            "text_embed_dim",
            "learned_embed_dim",
            "rnn_type",
            "chain_embed_dim",
            "event_history_dim",
        ]
        arch_defaults = {"rnn_type": "none"}
        mismatches = []
        missing = []
        sentinel = object()
        for field in arch_fields:
            ckpt_val = getattr(ckpt_config, field, sentinel)
            cli_val = getattr(config, field)
            if ckpt_val is sentinel:
                if field in arch_defaults:
                    ckpt_val = arch_defaults[field]
                else:
                    missing.append(field)
                    continue
            if ckpt_val != cli_val:
                mismatches.append(f"  {field}: checkpoint={ckpt_val}, cli={cli_val}")

        # rnn_hidden_dim and rnn_num_layers only shape weights when both
        # sides actually instantiate an RNN module.  When either side is
        # rnn_type="none" those values are placeholders that never touch
        # the state dict — comparing them would reject feed-forward
        # checkpoints whose saved placeholders drift from CLI defaults.
        # If rnn_type itself mismatches, the check above already fires.
        ckpt_rnn_type = getattr(ckpt_config, "rnn_type", "none")
        if config.is_recurrent and ckpt_rnn_type != "none":
            for field in ("rnn_hidden_dim", "rnn_num_layers"):
                ckpt_val = getattr(ckpt_config, field, None)
                cli_val = getattr(config, field)
                if ckpt_val is None:
                    missing.append(field)
                elif ckpt_val != cli_val:
                    mismatches.append(f"  {field}: checkpoint={ckpt_val}, cli={cli_val}")
        if missing:
            logger.warning(
                "Checkpoint config missing fields (older version?): %s",
                ", ".join(missing),
            )
        if mismatches:
            raise ValueError(
                "Architecture mismatch between checkpoint and CLI config:\n" + "\n".join(mismatches)
            )

        # deck_paths must match: the index-keyed _deck_wins map was saved
        # using the checkpoint's deck_paths ordering; resuming with a
        # different list silently attributes wins to the wrong deck names.
        # CLI resumes load deck_paths from ckpt["config"] before constructing
        # the trainer, so this check is defensive for direct callers who
        # build TrainingConfig(resume_checkpoint=...) without first merging.
        ckpt_decks = getattr(ckpt_config, "deck_paths", None)
        if ckpt_decks is not None and list(ckpt_decks) != list(config.deck_paths):
            raise ValueError(
                "deck_paths mismatch between checkpoint and config "
                "(index-keyed per-deck metrics would misalign):\n"
                f"  checkpoint: {list(ckpt_decks)}\n"
                f"  config:     {list(config.deck_paths)}"
            )

        # Text embedding mode: from_state_dict auto-detects from keys,
        # but warn if modes disagree so user is aware
        ckpt_has_text = any(k.startswith("text_lookup.") for k in ckpt["model_state_dict"])
        cli_wants_text = bool(config.card_embeddings)
        if ckpt_has_text and not cli_wants_text:
            logger.warning(
                "Checkpoint was trained with text embeddings but --card-embeddings "
                "not specified. Text embedding layers will be loaded from checkpoint."
            )
        elif cli_wants_text and not ckpt_has_text:
            raise ValueError(
                "CLI specifies --card-embeddings but checkpoint has no text embedding "
                "layers. Cannot add text embeddings to a symbolic-mode checkpoint."
            )

    def train(self) -> None:
        """Run the full PPO training loop."""
        config = self.config
        num_updates = config.total_timesteps // (config.rollout_steps * config.num_envs)

        if self._resume_update >= num_updates:
            logger.warning(
                "Resume update %d >= total updates %d — training already complete",
                self._resume_update,
                num_updates,
            )
            return

        logger.info(
            "Starting training: %d timesteps, %d updates, %d envs",
            config.total_timesteps,
            num_updates,
            config.num_envs,
        )

        pool_handles = (
            self._opponent_pool.share_handles() if self._opponent_pool is not None else None
        )

        if config.vec_env_type == "async_actor_learner":
            from yugioh_rl.actor_learner import AsyncActorLearnerVecEnv

            vec_env = AsyncActorLearnerVecEnv(
                num_envs=config.num_envs,
                deck_pool=self._deck_pool,
                opponent=config.opponent,
                reward_shaping=config.reward_shaping,
                shaping_lp_weight=config.shaping_lp_weight,
                shaping_card_weight=config.shaping_card_weight,
                seed=config.seed,
                agent_player=config.agent_player,
                deck_allocation=config.deck_allocation,
                mirror_decks=config.mirror_decks,
                opponent_device=None,
                master_model=self.network,
                config=config,
                rollout_steps=config.rollout_steps,
                opponent_pool_handles=pool_handles,
                opponent_pool_temperature=config.self_play_temperature,
                opponent_pool_sampling=config.self_play_sampling,
                opponent_pool_config=config,
            )
        elif config.vec_env_type == "sync_actor_learner":
            from yugioh_rl.actor_learner import ActorLearnerVecEnv

            vec_env = ActorLearnerVecEnv(
                num_envs=config.num_envs,
                deck_pool=self._deck_pool,
                opponent=config.opponent,
                reward_shaping=config.reward_shaping,
                shaping_lp_weight=config.shaping_lp_weight,
                shaping_card_weight=config.shaping_card_weight,
                seed=config.seed,
                agent_player=config.agent_player,
                deck_allocation=config.deck_allocation,
                mirror_decks=config.mirror_decks,
                opponent_device=None,  # workers use TrainingEnv's default resolution
                master_model=self.network,
                config=config,
                rollout_steps=config.rollout_steps,
                opponent_pool_handles=pool_handles,
                opponent_pool_temperature=config.self_play_temperature,
                opponent_pool_sampling=config.self_play_sampling,
                opponent_pool_config=config,
            )
        else:
            vec_env = SubprocVecEnv(
                num_envs=config.num_envs,
                deck_pool=self._deck_pool,
                opponent=config.opponent,
                reward_shaping=config.reward_shaping,
                shaping_lp_weight=config.shaping_lp_weight,
                shaping_card_weight=config.shaping_card_weight,
                seed=config.seed,
                agent_player=config.agent_player,
                deck_allocation=config.deck_allocation,
                mirror_decks=config.mirror_decks,
                master_model=self.network,
                rollout_steps=config.rollout_steps,
                opponent_pool_handles=pool_handles,
                opponent_pool_temperature=config.self_play_temperature,
                opponent_pool_sampling=config.self_play_sampling,
                opponent_pool_config=config,
            )

        try:
            global_step = self._resume_global_step
            start_time = time.time()

            for update in range(self._resume_update + 1, num_updates + 1):
                self.buffer.reset()

                # hx_initial for TBPTT replay — zero hx matches rollout
                # boundaries (actor-learner workers and SubprocVecEnv both
                # reset hx at the start of each rollout).
                self.buffer.hx_initial = self.network.init_hx(config.num_envs, self.device)

                # --- Collect rollout ---
                async_discarded = 0
                async_version_lags: list[int] = []
                if config.vec_env_type == "async_actor_learner":
                    rollouts, async_discarded, async_version_lags = vec_env.collect_rollouts(
                        config.max_version_lag
                    )
                else:
                    rollouts = vec_env.collect_rollouts()
                obs = self.buffer.ingest_rollouts(rollouts)
                for r in rollouts:
                    for info in r["infos"]:
                        self._record_episode(info)
                global_step += config.num_envs * config.rollout_steps

                # --- Compute advantages ---
                bootstrap_hx = self.network.cat_hx(
                    [r["final_hx"] for r in rollouts],
                    self.device,
                )
                with torch.no_grad():
                    t_cards = torch.from_numpy(obs["cards"]).to(self.device)
                    t_global = torch.from_numpy(obs["global_state"]).to(self.device)
                    t_actions = torch.from_numpy(obs["actions"]).to(self.device)
                    t_mask = torch.from_numpy(obs["action_mask"]).to(self.device)
                    t_chain = torch.from_numpy(obs["pending_chain"]).to(self.device)
                    t_event = torch.from_numpy(obs["event_history"]).to(self.device)
                    _, last_values, _ = self.network(
                        t_cards,
                        t_global,
                        t_actions,
                        t_mask,
                        hx=bootstrap_hx,
                        obs_chain=t_chain,
                        obs_event=t_event,
                    )
                    last_values_np = last_values.cpu().numpy()

                if config.vec_env_type == "async_actor_learner":
                    from yugioh_rl.vtrace import compute_vtrace

                    T, N = config.rollout_steps, config.num_envs
                    with torch.no_grad():
                        all_obs_cards = _to_tensor(self.buffer.obs_cards[:T], self.device)
                        all_obs_global = _to_tensor(self.buffer.obs_global[:T], self.device)
                        all_obs_actions = _to_tensor(self.buffer.obs_actions[:T], self.device)
                        all_obs_mask = _to_tensor(self.buffer.obs_mask[:T], self.device)
                        all_obs_chain = _to_tensor(self.buffer.obs_chain[:T], self.device)
                        all_obs_event = _to_tensor(self.buffer.obs_event[:T], self.device)
                        all_actions = _to_tensor(self.buffer.actions[:T], self.device, long=True)
                        all_log_probs_old = _to_tensor(self.buffer.log_probs[:T], self.device)

                        flat_cards = all_obs_cards.reshape(T * N, *all_obs_cards.shape[2:])
                        flat_global = all_obs_global.reshape(T * N, *all_obs_global.shape[2:])
                        flat_actions_obs = all_obs_actions.reshape(
                            T * N, *all_obs_actions.shape[2:]
                        )
                        flat_mask = all_obs_mask.reshape(T * N, *all_obs_mask.shape[2:])
                        flat_chain = all_obs_chain.reshape(T * N, *all_obs_chain.shape[2:])
                        flat_event = all_obs_event.reshape(T * N, *all_obs_event.shape[2:])
                        flat_acts = all_actions.reshape(T * N)

                        logits_new, values_new, _ = self.network(
                            flat_cards,
                            flat_global,
                            flat_actions_obs,
                            flat_mask,
                            obs_chain=flat_chain,
                            obs_event=flat_event,
                        )
                        dist_new = Categorical(logits=logits_new)
                        log_probs_new = dist_new.log_prob(flat_acts)

                        log_probs_new = log_probs_new.reshape(T, N)
                        values_new = values_new.reshape(T, N)

                    advantages, returns = compute_vtrace(
                        log_probs_old=all_log_probs_old,
                        log_probs_new=log_probs_new,
                        values=values_new,
                        rewards=_to_tensor(self.buffer.rewards[:T], self.device),
                        dones=_to_tensor(self.buffer.dones[:T], self.device),
                        last_values=_to_tensor(last_values_np, self.device),
                        gamma=config.gamma,
                        rho_bar=config.vtrace_rho_bar,
                        c_bar=config.vtrace_c_bar,
                    )
                    self.buffer.advantages[:T] = advantages.cpu().numpy()
                    self.buffer.returns[:T] = returns.cpu().numpy()
                else:
                    self.buffer.compute_advantages(last_values_np, config.gamma, config.gae_lambda)

                # --- PPO update ---
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0
                num_batches = 0

                if config.is_recurrent:
                    update_stats = self._run_update_tbptt()
                else:
                    update_stats = self._run_update_feedforward()
                total_policy_loss, total_value_loss, total_entropy, num_batches = update_stats

                # Publish fresh weights so workers (actor-learner) or the
                # local model ref (subproc) see them at the next rollout.
                vec_env.publish_weights(self.network)

                # --- Logging ---
                if update % config.log_interval == 0 and num_batches > 0:
                    elapsed = time.time() - start_time
                    fps = (global_step - self._resume_global_step) / elapsed

                    avg_policy_loss = total_policy_loss / num_batches
                    avg_value_loss = total_value_loss / num_batches
                    avg_entropy = total_entropy / num_batches

                    log_parts = [
                        f"Update {update}/{num_updates}",
                        f"steps={global_step}",
                        f"FPS={fps:.0f}",
                        f"policy_loss={avg_policy_loss:.4f}",
                        f"value_loss={avg_value_loss:.4f}",
                        f"entropy={avg_entropy:.4f}",
                    ]

                    if self._episode_rewards:
                        recent = self._episode_rewards[-100:]
                        recent_wins = self._episode_wins[-100:]
                        recent_lens = self._episode_lengths[-100:]
                        log_parts.extend(
                            [
                                f"ep_reward={np.mean(recent):.3f}",
                                f"win_rate={np.mean(recent_wins):.3f}",
                                f"ep_len={np.mean(recent_lens):.0f}",
                                f"episodes={len(self._episode_rewards)}",
                            ]
                        )

                    logger.info(" | ".join(log_parts))

                    if self._writer is not None:
                        self._writer.add_scalar("loss/policy", avg_policy_loss, global_step)
                        self._writer.add_scalar("loss/value", avg_value_loss, global_step)
                        self._writer.add_scalar("loss/entropy", avg_entropy, global_step)
                        self._writer.add_scalar("perf/fps", fps, global_step)
                        if self._episode_rewards:
                            self._writer.add_scalar("episode/reward", np.mean(recent), global_step)
                            self._writer.add_scalar(
                                "episode/win_rate", np.mean(recent_wins), global_step
                            )
                            self._writer.add_scalar(
                                "episode/length", np.mean(recent_lens), global_step
                            )
                        for deck_idx, wins_list in self._deck_wins.items():
                            if wins_list:
                                deck_name = Path(self.config.deck_paths[deck_idx]).stem
                                self._writer.add_scalar(
                                    f"episode/win_rate_deck_{deck_name}",
                                    np.mean(wins_list[-100:]),
                                    global_step,
                                )
                        if self._opponent_pool is not None:
                            elo = self._opponent_pool.elo_summary()
                            self._writer.add_scalar("selfplay/elo_agent", elo["agent"], global_step)
                            self._writer.add_scalar(
                                "selfplay/elo_pool_mean", elo["pool_mean"], global_step
                            )
                            self._writer.add_scalar(
                                "selfplay/elo_pool_min", elo["pool_min"], global_step
                            )
                            self._writer.add_scalar(
                                "selfplay/elo_pool_max", elo["pool_max"], global_step
                            )
                            self._writer.add_scalar(
                                "selfplay/occupied", elo["occupied"], global_step
                            )
                        if config.vec_env_type == "async_actor_learner":
                            self._writer.add_scalar(
                                "async/version_lag_mean",
                                np.mean(async_version_lags) if async_version_lags else 0,
                                global_step,
                            )
                            self._writer.add_scalar(
                                "async/rollouts_discarded",
                                async_discarded,
                                global_step,
                            )
                            depth = vec_env.queue_depth
                            if depth is not None:
                                self._writer.add_scalar("async/queue_depth", depth, global_step)

                # --- Evaluation ---
                if update % config.eval_interval == 0:
                    self._evaluate(config.eval_episodes, global_step)

                # --- Checkpointing ---
                if update % config.save_interval == 0:
                    self._save_checkpoint(update, global_step)

            # --- Final checkpoint (if not already saved at this update) ---
            if num_updates % config.save_interval != 0:
                self._save_checkpoint(num_updates, global_step)

        finally:
            vec_env.close()
            if self._writer is not None:
                self._writer.close()

        logger.info("Training complete. Total steps: %d", global_step)

    def _record_episode(self, info: dict) -> None:
        """Append a completed-episode entry to the trainer-side counters.

        Both the subproc and actor-learner branches surface terminations via
        an info dict with ``terminal_reward`` set; this consolidates the
        bookkeeping so the two paths stay in sync.
        """
        if "terminal_reward" not in info:
            return
        self._episode_rewards.append(info["terminal_reward"])
        self._episode_lengths.append(info.get("episode_length", 0))
        win = 1.0 if info["terminal_reward"] > 0 else 0.0
        self._episode_wins.append(win)
        if "agent_deck_idx" in info:
            self._deck_wins.setdefault(info["agent_deck_idx"], []).append(win)

    def _ppo_loss_terms_unreduced(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample policy / value / entropy-loss terms (no reduction).

        Returns ``(pg, v, ent_loss, entropy)``.  Caller .mean()s the first
        three exactly once per minibatch — concatenating across TBPTT chunks
        first if needed — so gradient magnitude matches a single .mean()
        across the full sample budget.  ``entropy`` is for logging only.
        """
        config = self.config
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
        pg = -torch.min(surr1, surr2)
        v = (values - returns) ** 2
        ent_loss = -entropy
        return pg, v, ent_loss, entropy

    def _step_optimizer(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> None:
        """Combine the three reduced losses, backward, clip, step."""
        config = self.config
        loss = (
            policy_loss + config.value_loss_coef * value_loss + config.entropy_coef * entropy_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.max_grad_norm)
        self.optimizer.step()

    def _run_update_feedforward(self) -> tuple[float, float, float, int]:
        """Feed-forward PPO update (default; ``rnn_type == 'none'``)."""
        config = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(config.num_epochs):
            for batch in self.buffer.get_batches(config.minibatch_size, self.device):
                logits, values, _ = self.network(
                    batch.obs_cards,
                    batch.obs_global,
                    batch.obs_actions,
                    batch.action_mask,
                    obs_chain=batch.obs_chain,
                    obs_event=batch.obs_event,
                )
                pg, v, ent_loss, entropy = self._ppo_loss_terms_unreduced(
                    logits,
                    values,
                    batch.actions,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.returns,
                )
                policy_loss = pg.mean()
                value_loss = v.mean()
                entropy_loss = ent_loss.mean()

                self._step_optimizer(policy_loss, value_loss, entropy_loss)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        return total_policy_loss, total_value_loss, total_entropy, num_batches

    def _run_update_tbptt(self) -> tuple[float, float, float, int]:
        """TBPTT PPO update for recurrent policies.

        Each minibatch is one (T, env_mb, ...) slice of the rollout.  The
        chunk loop walks T in steps of L = ``bptt_chunk_len``, threading
        hx with ``detach()`` between chunks so backprop is bounded to L.

        ``backward()`` runs PER chunk so the autograd graph for each
        chunk is freed before the next forward — peak activation memory
        is bounded by ``L * env_mb``, not ``T * env_mb``.  Each chunk's
        loss is scaled by ``L / T`` so the gradient accumulated into
        ``param.grad`` over all chunks equals a single ``.mean()`` over
        ``T * env_mb`` samples in the feed-forward path.
        """
        config = self.config
        T = config.rollout_steps
        L = config.bptt_chunk_len
        scale = L / T

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(config.num_epochs):
            for batch in self.buffer.get_recurrent_batches(
                config.minibatch_size,
                self.device,
            ):
                env_mb = batch.obs_cards.shape[1]
                self.optimizer.zero_grad()

                hx = batch.hx_initial
                # Accumulate metrics as device tensors so we sync once per
                # minibatch instead of 3× per chunk.  At T=256 / L=16 that's
                # the difference between ~50 and ~1500 device-host syncs per
                # PPO update.
                mb_policy_loss = torch.zeros((), device=self.device)
                mb_value_loss = torch.zeros((), device=self.device)
                mb_entropy = torch.zeros((), device=self.device)

                for chunk_start in range(0, T, L):
                    chunk = slice(chunk_start, chunk_start + L)
                    flat = L * env_mb

                    logits, values, hx_new = self.network(
                        batch.obs_cards[chunk].reshape(flat, MAX_CARDS, CARD_FEATURES),
                        batch.obs_global[chunk].reshape(flat, GLOBAL_FEATURES),
                        batch.obs_actions[chunk].reshape(flat, MAX_ACTIONS, ACTION_FEATURES),
                        batch.action_mask[chunk].reshape(flat, MAX_ACTIONS),
                        hx=hx,
                        seq_shape=(L, env_mb),
                        dones=batch.dones[chunk],
                        obs_chain=batch.obs_chain[chunk].reshape(
                            flat, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES
                        ),
                        obs_event=batch.obs_event[chunk].reshape(
                            flat, MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES
                        ),
                    )

                    pg, v, ent_loss, entropy = self._ppo_loss_terms_unreduced(
                        logits,
                        values,
                        batch.actions[chunk].reshape(flat),
                        batch.old_log_probs[chunk].reshape(flat),
                        batch.advantages[chunk].reshape(flat),
                        batch.returns[chunk].reshape(flat),
                    )

                    chunk_loss = (
                        pg.mean()
                        + config.value_loss_coef * v.mean()
                        + config.entropy_coef * ent_loss.mean()
                    ) * scale
                    chunk_loss.backward()

                    hx = YuGiOhNet.detach_hx(hx_new)

                    # Detach so the metric tensors don't hold this chunk's
                    # autograd graph past .backward().
                    mb_policy_loss += pg.detach().mean() * scale
                    mb_value_loss += v.detach().mean() * scale
                    mb_entropy += entropy.detach().mean() * scale

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += mb_policy_loss.item()
                total_value_loss += mb_value_loss.item()
                total_entropy += mb_entropy.item()
                num_batches += 1

        return total_policy_loss, total_value_loss, total_entropy, num_batches

    def _evaluate(self, num_episodes: int, global_step: int) -> None:
        """Evaluate the live network against configured opponents."""
        self.network.eval()
        try:
            agent = NetworkOpponent(self.network, device=str(self.device))
            # opponent_device left None so YUGIOH_OPPONENT_DEVICE / "cpu" default
            # still wins for eval-side model opponents. Forcing the trainer's
            # GPU here would silently override an explicit env-var opt-out and
            # can OOM on memory-constrained GPUs.
            results = evaluate_with_agent(
                agent,
                deck_pool=self._deck_pool,
                opponent_specs=self.config.eval_opponents,
                num_episodes=num_episodes,
                seed=self.config.seed + 999999,
                agent_player=self.config.agent_player,
            )
            for r in results:
                logger.info(
                    "Eval vs %s: %d/%d wins (%.1f%%)",
                    r.opponent_label,
                    r.wins,
                    r.episodes,
                    r.win_rate * 100,
                )
                for deck_idx, deck_results in r.per_deck_wins.items():
                    deck_name = Path(self.config.deck_paths[deck_idx]).stem
                    logger.info(
                        "  deck %s: %d/%d wins (%.1f%%)",
                        deck_name,
                        int(sum(deck_results)),
                        len(deck_results),
                        float(np.mean(deck_results)) * 100,
                    )
            if self._writer is not None:
                log_results_to_tensorboard(
                    self._writer,
                    results,
                    self.config.deck_paths,
                    global_step,
                )
        finally:
            self.network.train()

    def _save_checkpoint(self, update: int, global_step: int) -> None:
        """Save model checkpoint and update latest.pt symlink."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"checkpoint_{update}.pt"

        torch.save(
            {
                "update": update,
                "global_step": global_step,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "episode_rewards": self._episode_rewards[-1000:],
                "episode_lengths": self._episode_lengths[-1000:],
                "episode_wins": self._episode_wins[-1000:],
                "deck_wins": {k: v[-1000:] for k, v in self._deck_wins.items()},
            },
            path,
        )

        latest = save_dir / "checkpoint_latest.pt"
        latest.unlink(missing_ok=True)
        latest.symlink_to(path.name)

        logger.info("Saved checkpoint to %s", path)

        if self._opponent_pool is not None:
            self._opponent_pool.add_snapshot(self.network)
