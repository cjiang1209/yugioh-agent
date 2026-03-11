"""Policy and value network for Yu-Gi-Oh! PPO agent."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)

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


class TextEmbeddingLookup(nn.Module):
    """Frozen sentence-transformer embeddings with trainable projection.

    Loads pre-computed card text embeddings from a .pt file and provides
    vectorized lookup by card passcode using torch.searchsorted.

    The frozen embeddings are projected through a trainable Linear layer
    to produce the final text representation.
    """

    def __init__(self, embeddings_path: str, text_embed_dim: int) -> None:
        super().__init__()

        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        codes = data["codes"]  # (N,) int64
        embeddings = data["embeddings"]  # (N, raw_dim) float32

        # Sort by code for searchsorted
        sorted_indices = codes.argsort()
        sorted_codes = codes[sorted_indices]
        sorted_embeddings = embeddings[sorted_indices]

        # Prepend zero row at index 0 (for unknown/padding cards)
        raw_dim = sorted_embeddings.shape[1]
        padded_embeddings = torch.cat(
            [torch.zeros(1, raw_dim), sorted_embeddings], dim=0
        )  # (N+1, raw_dim)

        self.register_buffer("_sorted_codes", sorted_codes)

        self._frozen_embed = nn.Embedding.from_pretrained(
            padded_embeddings, freeze=True, padding_idx=0
        )

        self._proj = nn.Linear(raw_dim, text_embed_dim)

        logger.info(
            "TextEmbeddingLookup: %d cards, raw_dim=%d, proj_dim=%d",
            len(sorted_codes), raw_dim, text_embed_dim,
        )

    @property
    def num_cards(self) -> int:
        """Number of known cards (excluding padding)."""
        return len(self._sorted_codes)

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up text embeddings for card codes.

        Args:
            codes: (...) long tensor of card passcodes

        Returns:
            text_embed: (..., text_embed_dim) float — projected text embeddings
            embed_idx: (...) long — indices into the (N+1)-row embedding table
                       (0 = unknown/padding, 1..N = known cards)
        """
        idx = torch.searchsorted(self._sorted_codes, codes)
        idx = idx.clamp(0, len(self._sorted_codes) - 1)
        valid = self._sorted_codes[idx] == codes
        # Shift by 1 because row 0 is padding; unknown codes map to 0
        embed_idx = torch.where(valid, idx + 1, torch.zeros_like(idx))

        frozen = self._frozen_embed(embed_idx)
        return self._proj(frozen), embed_idx


class YuGiOhNet(nn.Module):
    """Combined policy + value network for Yu-Gi-Oh! RL.

    Architecture:
        1. Card encoder: embedding + MLP per card → (B, 200, card_embed_dim)
        2. Zone pooling: mean-pool cards by (location, controller) → (B, 12, card_embed_dim)
        3. Global encoder: MLP on global features
        4. Board representation: MLP on concat(zone_pool_flat, global)
        5. Policy head: dot-product scoring of action embeddings vs board projection
        6. Value head: MLP on board representation → scalar

    Two card embedding modes:
        Symbolic (default): cards are arbitrary tokens — modulo-hashed into a
            fixed-size learned embedding with no built-in knowledge of card effects.
        Semantic (--card-embeddings): cards carry meaning — frozen sentence-transformer
            text embeddings (projected) + collision-free learned embedding.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()

        self._use_text_embeddings = bool(config.card_embeddings_path)

        if self._use_text_embeddings:
            # Semantic mode
            self.text_lookup = TextEmbeddingLookup(
                config.card_embeddings_path, config.text_embed_dim
            )
            num_entries = self.text_lookup.num_cards + 1  # +1 for padding at 0
            self.card_embedding = nn.Embedding(
                num_entries, config.learned_embed_dim, padding_idx=0
            )
            embed_dim = config.text_embed_dim + config.learned_embed_dim
        else:
            # Symbolic mode
            self.text_lookup = None
            self.card_embedding = nn.Embedding(
                _CARD_VOCAB, _CARD_EMBED_DIM, padding_idx=0
            )
            embed_dim = _CARD_EMBED_DIM

        # Card encoder
        card_input_dim = embed_dim + CARD_FEAT_DIM
        self.card_encoder = _mlp(card_input_dim, 128, config.card_embed_dim)

        # Global encoder
        self.global_encoder = _mlp(GLOBAL_FEAT_DIM, config.global_embed_dim, config.global_embed_dim)

        # Board representation
        board_input_dim = _NUM_ZONES * config.card_embed_dim + config.global_embed_dim
        self.board_mlp = _mlp(board_input_dim, config.board_hidden_dim, config.board_hidden_dim)

        # Action encoder
        action_input_dim = embed_dim + ACTION_FEAT_DIM
        self.action_encoder = _mlp(action_input_dim, config.action_embed_dim, config.action_embed_dim)

        # Policy head: project board → action_embed_dim for dot product
        self.board_proj = nn.Linear(config.board_hidden_dim, config.action_embed_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.board_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed card codes using symbolic or semantic mode.

        Args:
            codes: (...) long tensor of card passcodes

        Returns:
            (..., embed_dim) float tensor
        """
        if self.text_lookup is not None:
            text_embed, embed_idx = self.text_lookup(codes)
            learned_embed = self.card_embedding(embed_idx)
            return torch.cat([text_embed, learned_embed], dim=-1)
        else:
            return self.card_embedding(codes % _CARD_VOCAB)

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
        card_embed = self._embed_codes(card_ids)  # (B, 200, embed_dim)
        card_input = torch.cat([card_embed, card_feats], dim=-1)
        card_enc = self.card_encoder(card_input)  # (B, 200, card_embed_dim)

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
        act_embed = self._embed_codes(action_codes)  # (B, 32, embed_dim)
        act_input = torch.cat([act_embed, action_feats], dim=-1)
        act_enc = self.action_encoder(act_input)  # (B, 32, action_embed_dim)

        # --- Policy head: dot product ---
        board_p = self.board_proj(board)  # (B, action_embed_dim)
        logits = (act_enc * board_p.unsqueeze(1)).sum(dim=-1)  # (B, 32)

        # Mask illegal actions
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # --- Value head ---
        values = self.value_head(board).squeeze(-1)  # (B,)

        return logits, values
