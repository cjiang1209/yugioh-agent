"""PPO algorithm with rollout buffer for Yu-Gi-Oh! training."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from yugioh_env.opponent import NetworkOpponent
from yugioh_rl.config import TrainingConfig
from yugioh_rl.env_wrapper import SubprocVecEnv, TrainingEnv, parse_deck_pool
from yugioh_rl.eval import evaluate, log_results_to_tensorboard
from yugioh_rl.network import YuGiOhNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class MiniBatch:
    """A single minibatch of training data."""

    obs_cards: torch.Tensor      # (M, 200, 42)
    obs_global: torch.Tensor     # (M, 20)
    obs_actions: torch.Tensor    # (M, 32, 12)
    action_mask: torch.Tensor    # (M, 32)
    actions: torch.Tensor        # (M,)
    old_log_probs: torch.Tensor  # (M,)
    advantages: torch.Tensor     # (M,)
    returns: torch.Tensor        # (M,)


class RolloutBuffer:
    """Stores trajectory data for PPO rollouts."""

    def __init__(self, rollout_steps: int, num_envs: int) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self._ptr = 0

        T, N = rollout_steps, num_envs
        self.obs_cards = np.zeros((T, N, 200, 42), dtype=np.uint8)
        self.obs_global = np.zeros((T, N, 20), dtype=np.uint8)
        self.obs_actions = np.zeros((T, N, 32, 12), dtype=np.uint8)
        self.obs_mask = np.zeros((T, N, 32), dtype=np.int8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    def reset(self) -> None:
        self._ptr = 0

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
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self._ptr += 1

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
        flat_cards = self.obs_cards.reshape(total, 200, 42)
        flat_global = self.obs_global.reshape(total, 20)
        flat_actions_obs = self.obs_actions.reshape(total, 32, 12)
        flat_mask = self.obs_mask.reshape(total, 32)
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
                obs_cards=torch.from_numpy(flat_cards[idx]).to(device),
                obs_global=torch.from_numpy(flat_global[idx]).to(device),
                obs_actions=torch.from_numpy(flat_actions_obs[idx]).to(device),
                action_mask=torch.from_numpy(flat_mask[idx]).to(device),
                actions=torch.from_numpy(flat_actions[idx]).long().to(device),
                old_log_probs=torch.from_numpy(flat_log_probs[idx]).to(device),
                advantages=torch.from_numpy(flat_advantages[idx]).to(device),
                returns=torch.from_numpy(flat_returns[idx]).to(device),
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
            self.network = YuGiOhNet.from_state_dict(
                config, ckpt["model_state_dict"]
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=config.learning_rate
            )
            if config.resume_optimizer and "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Override LR from CLI so users can change schedule across runs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = config.learning_rate
            logger.info("Initialized weights from checkpoint: %s", config.init_checkpoint)
        else:
            self.network = YuGiOhNet.from_config(config).to(self.device)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

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

        self.network = YuGiOhNet.from_state_dict(
            config, ckpt["model_state_dict"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
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
        self._deck_wins = {
            int(k): list(v) for k, v in ckpt.get("deck_wins", {}).items()
        }

        update = ckpt.get("update", 0)
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Loaded checkpoint for resumption: %s (update=%d, global_step=%d)",
            config.resume_checkpoint, update, global_step,
        )
        return update, global_step

    @staticmethod
    def _validate_checkpoint_compat(config: TrainingConfig, ckpt: dict) -> None:
        """Validate architecture compatibility before loading weights."""
        ckpt_config = ckpt.get("config")
        if ckpt_config is None:
            logger.warning("Checkpoint has no saved config — skipping compatibility check")
            return

        # Architecture fields that determine layer shapes — must match exactly
        arch_fields = [
            "card_embed_dim", "global_embed_dim", "board_hidden_dim",
            "action_embed_dim", "text_embed_dim", "learned_embed_dim",
        ]
        mismatches = []
        missing = []
        for field in arch_fields:
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
                "Architecture mismatch between checkpoint and CLI config:\n"
                + "\n".join(mismatches)
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
        ckpt_has_text = any(
            k.startswith("text_lookup.") for k in ckpt["model_state_dict"]
        )
        cli_wants_text = bool(config.card_embeddings_path)
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
                self._resume_update, num_updates,
            )
            return

        logger.info(
            "Starting training: %d timesteps, %d updates, %d envs",
            config.total_timesteps, num_updates, config.num_envs,
        )

        vec_env = SubprocVecEnv(
            num_envs=config.num_envs,
            deck_pool=self._deck_pool,
            opponent=config.opponent,
            reward_shaping=config.reward_shaping,
            shaping_lp_weight=config.shaping_lp_weight,
            shaping_card_weight=config.shaping_card_weight,
            seed=config.seed,
            agent_player=config.agent_player,
        )

        try:
            obs = vec_env.reset()
            global_step = self._resume_global_step
            start_time = time.time()

            for update in range(self._resume_update + 1, num_updates + 1):
                update_start = time.time()
                self.buffer.reset()

                # --- Collect rollout ---
                for step in range(config.rollout_steps):
                    with torch.no_grad():
                        t_cards = torch.from_numpy(obs["cards"]).to(self.device)
                        t_global = torch.from_numpy(obs["global_state"]).to(self.device)
                        t_actions = torch.from_numpy(obs["actions"]).to(self.device)
                        t_mask = torch.from_numpy(obs["action_mask"]).to(self.device)

                        logits, values = self.network(t_cards, t_global, t_actions, t_mask)
                        dist = Categorical(logits=logits)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

                    actions_np = actions.cpu().numpy()
                    log_probs_np = log_probs.cpu().numpy()
                    values_np = values.cpu().numpy()

                    next_obs, rewards, dones, infos = vec_env.step(actions_np)

                    self.buffer.add(obs, actions_np, log_probs_np, rewards, dones.astype(np.float32), values_np)

                    # Track completed episodes
                    for info in infos:
                        if "terminal_reward" in info:
                            self._episode_rewards.append(info["terminal_reward"])
                            self._episode_lengths.append(info.get("episode_length", 0))
                            win = 1.0 if info["terminal_reward"] > 0 else 0.0
                            self._episode_wins.append(win)
                            if "agent_deck_idx" in info:
                                self._deck_wins.setdefault(info["agent_deck_idx"], []).append(win)

                    obs = next_obs
                    global_step += config.num_envs

                # --- Compute advantages ---
                with torch.no_grad():
                    t_cards = torch.from_numpy(obs["cards"]).to(self.device)
                    t_global = torch.from_numpy(obs["global_state"]).to(self.device)
                    t_actions = torch.from_numpy(obs["actions"]).to(self.device)
                    t_mask = torch.from_numpy(obs["action_mask"]).to(self.device)
                    _, last_values = self.network(t_cards, t_global, t_actions, t_mask)
                    last_values_np = last_values.cpu().numpy()

                self.buffer.compute_advantages(last_values_np, config.gamma, config.gae_lambda)

                # --- PPO update ---
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0
                num_batches = 0

                for epoch in range(config.num_epochs):
                    for batch in self.buffer.get_batches(config.minibatch_size, self.device):
                        logits, values = self.network(
                            batch.obs_cards, batch.obs_global,
                            batch.obs_actions, batch.action_mask,
                        )
                        dist = Categorical(logits=logits)
                        log_probs = dist.log_prob(batch.actions)
                        entropy = dist.entropy()

                        # Clipped surrogate objective
                        ratio = (log_probs - batch.old_log_probs).exp()
                        surr1 = ratio * batch.advantages
                        surr2 = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * batch.advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = F.mse_loss(values, batch.returns)

                        # Entropy bonus
                        entropy_loss = -entropy.mean()

                        loss = (
                            policy_loss
                            + config.value_loss_coef * value_loss
                            + config.entropy_coef * entropy_loss
                        )

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), config.max_grad_norm)
                        self.optimizer.step()

                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        total_entropy += entropy.mean().item()
                        num_batches += 1

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
                        log_parts.extend([
                            f"ep_reward={np.mean(recent):.3f}",
                            f"win_rate={np.mean(recent_wins):.3f}",
                            f"ep_len={np.mean(recent_lens):.0f}",
                            f"episodes={len(self._episode_rewards)}",
                        ])

                    logger.info(" | ".join(log_parts))

                    if self._writer is not None:
                        self._writer.add_scalar("loss/policy", avg_policy_loss, global_step)
                        self._writer.add_scalar("loss/value", avg_value_loss, global_step)
                        self._writer.add_scalar("loss/entropy", avg_entropy, global_step)
                        self._writer.add_scalar("perf/fps", fps, global_step)
                        if self._episode_rewards:
                            self._writer.add_scalar("episode/reward", np.mean(recent), global_step)
                            self._writer.add_scalar("episode/win_rate", np.mean(recent_wins), global_step)
                            self._writer.add_scalar("episode/length", np.mean(recent_lens), global_step)
                        for deck_idx, wins_list in self._deck_wins.items():
                            if wins_list:
                                deck_name = Path(self.config.deck_paths[deck_idx]).stem
                                self._writer.add_scalar(
                                    f"episode/win_rate_deck_{deck_name}",
                                    np.mean(wins_list[-100:]),
                                    global_step,
                                )

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

    def _evaluate(self, num_episodes: int, global_step: int) -> None:
        """Evaluate the live network against configured opponents."""
        self.network.eval()
        try:
            agent = NetworkOpponent(self.network, device=str(self.device))
            # opponent_device left None so YUGIOH_OPPONENT_DEVICE / "cpu" default
            # still wins for eval-side model opponents. Forcing the trainer's
            # GPU here would silently override an explicit env-var opt-out and
            # can OOM on memory-constrained GPUs.
            results = evaluate(
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
                    r.opponent_label, r.wins, r.episodes, r.win_rate * 100,
                )
                for deck_idx, deck_results in r.per_deck_wins.items():
                    deck_name = Path(self.config.deck_paths[deck_idx]).stem
                    logger.info(
                        "  deck %s: %d/%d wins (%.1f%%)",
                        deck_name, int(sum(deck_results)), len(deck_results),
                        float(np.mean(deck_results)) * 100,
                    )
            if self._writer is not None:
                log_results_to_tensorboard(
                    self._writer, results, self.config.deck_paths, global_step,
                )
        finally:
            self.network.train()

    def _save_checkpoint(self, update: int, global_step: int) -> None:
        """Save model checkpoint and update latest.pt symlink."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"checkpoint_{update}.pt"

        torch.save({
            "update": update,
            "global_step": global_step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "episode_rewards": self._episode_rewards[-1000:],
            "episode_lengths": self._episode_lengths[-1000:],
            "episode_wins": self._episode_wins[-1000:],
            "deck_wins": {k: v[-1000:] for k, v in self._deck_wins.items()},
        }, path)

        latest = save_dir / "checkpoint_latest.pt"
        latest.unlink(missing_ok=True)
        latest.symlink_to(path.name)

        logger.info("Saved checkpoint to %s", path)
