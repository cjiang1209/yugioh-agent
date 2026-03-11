"""Policy and value network for Yu-Gi-Oh! PPO agent."""

from __future__ import annotations

import torch
import torch.nn as nn

from yugioh_rl.config import TrainingConfig
from yugioh_rl.features import (
    CARD_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    ACTION_FEAT_DIM,
    decode_cards,
    decode_global,
    decode_actions,
)

# Location bits used for zone pooling (same order as features.py _LOC_BITS)
# hand=0x02, mzone=0x04, szone=0x08, grave=0x10, banished=0x20, extra=0x40
_ZONE_LOC_BITS = [0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
_NUM_ZONES = len(_ZONE_LOC_BITS) * 2  # 6 zones × 2 players = 12

# Card embedding vocabulary (card codes mod-hashed from uint32)
_CARD_VOCAB = 131072
_CARD_EMBED_DIM = 16


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(),
    )


class YuGiOhNet(nn.Module):
    """Combined policy + value network for Yu-Gi-Oh! RL.

    Architecture:
        1. Card encoder: embedding + MLP per card → (B, 200, card_embed_dim)
        2. Zone pooling: mean-pool cards by (location, controller) → (B, 12, card_embed_dim)
        3. Global encoder: MLP on global features
        4. Board representation: MLP on concat(zone_pool_flat, global)
        5. Policy head: dot-product scoring of action embeddings vs board projection
        6. Value head: MLP on board representation → scalar
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()

        # Shared card embedding (used for both board cards and action cards)
        self.card_embedding = nn.Embedding(_CARD_VOCAB, _CARD_EMBED_DIM, padding_idx=0)

        # Card encoder
        card_input_dim = _CARD_EMBED_DIM + CARD_FEAT_DIM
        self.card_encoder = _mlp(card_input_dim, 128, config.card_embed_dim)

        # Global encoder
        self.global_encoder = _mlp(GLOBAL_FEAT_DIM, config.global_embed_dim, config.global_embed_dim)

        # Board representation
        board_input_dim = _NUM_ZONES * config.card_embed_dim + config.global_embed_dim
        self.board_mlp = _mlp(board_input_dim, config.board_hidden_dim, config.board_hidden_dim)

        # Action encoder
        action_input_dim = _CARD_EMBED_DIM + ACTION_FEAT_DIM
        self.action_encoder = _mlp(action_input_dim, config.action_embed_dim, config.action_embed_dim)

        # Policy head: project board → action_embed_dim for dot product
        self.board_proj = nn.Linear(config.board_hidden_dim, config.action_embed_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.board_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obs_cards: torch.Tensor,
        obs_global: torch.Tensor,
        obs_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing action logits and state value.

        Args:
            obs_cards: (B, 200, 42) uint8
            obs_global: (B, 20) uint8
            obs_actions: (B, 32, 12) uint8
            action_mask: (B, 32) int8 — 1=legal, 0=illegal

        Returns:
            logits: (B, 32) float — masked action logits
            values: (B,) float — state value estimates
        """
        # --- Decode observations ---
        card_ids, card_feats = decode_cards(obs_cards)      # (B,200), (B,200,F_card)
        global_feats = decode_global(obs_global)              # (B,F_global)
        action_codes, action_feats = decode_actions(obs_actions)  # (B,32), (B,32,F_act)

        # --- Card encoding ---
        card_embed = self.card_embedding(card_ids % _CARD_VOCAB)  # (B,200,16)
        card_input = torch.cat([card_embed, card_feats], dim=-1)  # (B,200,16+F_card)
        card_enc = self.card_encoder(card_input)  # (B,200,card_embed_dim)

        # --- Zone pooling ---
        # Extract raw location byte and controller byte for zone assignment
        raw_loc = obs_cards[..., 4].long()   # (B, 200)
        raw_ctrl = obs_cards[..., 7].long()  # (B, 200)

        zone_parts = []
        for ctrl in (0, 1):
            ctrl_mask = (raw_ctrl == ctrl)  # (B, 200)
            for bit in _ZONE_LOC_BITS:
                # Cards in this zone for this controller
                loc_mask = ((raw_loc & bit) != 0) & ctrl_mask  # (B, 200)
                # Masked mean pooling
                mask_f = loc_mask.float().unsqueeze(-1)  # (B, 200, 1)
                masked_sum = (card_enc * mask_f).sum(dim=1)  # (B, card_embed_dim)
                count = mask_f.sum(dim=1).clamp(min=1.0)      # (B, 1)
                zone_parts.append(masked_sum / count)           # (B, card_embed_dim)

        zone_flat = torch.cat(zone_parts, dim=-1)  # (B, 12*card_embed_dim)

        # --- Global encoding ---
        global_enc = self.global_encoder(global_feats)  # (B, global_embed_dim)

        # --- Board representation ---
        board_input = torch.cat([zone_flat, global_enc], dim=-1)
        board = self.board_mlp(board_input)  # (B, board_hidden_dim)

        # --- Action encoding ---
        act_embed = self.card_embedding(action_codes % _CARD_VOCAB)  # (B,32,16)
        act_input = torch.cat([act_embed, action_feats], dim=-1)  # (B,32,16+F_act)
        act_enc = self.action_encoder(act_input)  # (B, 32, action_embed_dim)

        # --- Policy head: dot product ---
        board_p = self.board_proj(board)  # (B, action_embed_dim)
        logits = (act_enc * board_p.unsqueeze(1)).sum(dim=-1)  # (B, 32)

        # Mask illegal actions
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # --- Value head ---
        values = self.value_head(board).squeeze(-1)  # (B,)

        return logits, values
