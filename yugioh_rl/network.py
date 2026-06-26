"""Policy and value network for Yu-Gi-Oh! PPO agent."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from yugioh_core.encoding import SYSSTRING_VOCAB
from yugioh_rl.config import TrainingConfig
from yugioh_rl.features import (
    ACTION_FEAT_DIM,
    CARD_FEAT_DIM,
    CHAIN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    decode_actions,
    decode_cards,
    decode_global,
    decode_pending_chain,
)

logger = logging.getLogger(__name__)

# Recurrent hidden state shape:
#   None              — feed-forward (rnn_type="none")
#   Tensor            — GRU (h, shape (num_layers, batch, hidden_dim))
#   tuple of 2 Tensor — LSTM ((h, c), each (num_layers, batch, hidden_dim))
HxState = tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None

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

    Provides vectorized lookup by card passcode using torch.searchsorted.
    The frozen embeddings are projected through a trainable Linear layer
    to produce the final text representation.

    Use ``from_path()`` at training time to load from a .pt file,
    or ``from_state_dict_shapes()`` to reconstruct from a checkpoint
    without any disk I/O.
    """

    def __init__(
        self, sorted_codes: torch.Tensor, padded_embeddings: torch.Tensor, text_embed_dim: int
    ) -> None:
        super().__init__()
        assert padded_embeddings.shape[0] == len(sorted_codes) + 1, (
            f"padded_embeddings row count ({padded_embeddings.shape[0]}) "
            f"must be len(sorted_codes) + 1 ({len(sorted_codes) + 1})"
        )
        self.register_buffer("_sorted_codes", sorted_codes)
        self._frozen_embed = nn.Embedding.from_pretrained(
            padded_embeddings, freeze=True, padding_idx=0
        )
        self._proj = nn.Linear(padded_embeddings.shape[1], text_embed_dim)

    @classmethod
    def from_path(cls, embeddings_path: str, text_embed_dim: int) -> TextEmbeddingLookup:
        """Load card text embeddings from a .pt file (training time)."""
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

        lookup = cls(sorted_codes, padded_embeddings, text_embed_dim)

        logger.info(
            "TextEmbeddingLookup: %d cards, raw_dim=%d, proj_dim=%d",
            len(sorted_codes),
            raw_dim,
            text_embed_dim,
        )
        return lookup

    @classmethod
    def from_state_dict_shapes(
        cls,
        text_embed_dim: int,
        state_dict: dict[str, torch.Tensor],
    ) -> TextEmbeddingLookup:
        """Build a correctly-shaped skeleton from state dict keys (no disk I/O).

        Creates zero-filled buffers/params matching the shapes in *state_dict*.
        The caller is responsible for calling ``load_state_dict()`` to fill in
        real values.
        """
        num_cards = state_dict["_sorted_codes"].shape[0]  # excludes padding row
        raw_dim = state_dict["_frozen_embed.weight"].shape[1]
        sorted_codes = torch.zeros(num_cards, dtype=torch.int64)
        # _frozen_embed.weight has num_cards+1 rows: row 0 is padding, rows 1..N are cards
        padded = torch.zeros(num_cards + 1, raw_dim)
        return cls(sorted_codes, padded, text_embed_dim)

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

    def __init__(
        self, config: TrainingConfig, text_lookup: TextEmbeddingLookup | None = None
    ) -> None:
        super().__init__()

        self._use_text_embeddings = text_lookup is not None

        if self._use_text_embeddings:
            # Semantic mode
            self.text_lookup = text_lookup
            num_entries = text_lookup.num_cards + 1  # +1 for padding at 0
            self.card_embedding = nn.Embedding(num_entries, config.learned_embed_dim, padding_idx=0)
            embed_dim = config.text_embed_dim + config.learned_embed_dim
        else:
            # Symbolic mode
            self.text_lookup = None
            self.card_embedding = nn.Embedding(_CARD_VOCAB, _CARD_EMBED_DIM, padding_idx=0)
            embed_dim = _CARD_EMBED_DIM

        # Card encoder
        card_input_dim = embed_dim + CARD_FEAT_DIM
        self.card_encoder = _mlp(card_input_dim, 128, config.card_embed_dim)

        # Global encoder
        self.global_encoder = _mlp(
            GLOBAL_FEAT_DIM, config.global_embed_dim, config.global_embed_dim
        )

        # ── Pending chain encoder (optional) ──
        self._chain_embed_dim = config.chain_embed_dim
        if config.chain_embed_dim > 0:
            chain_entry_dim = embed_dim + embed_dim + config.desc_n_embed_dim + CHAIN_FEAT_DIM
            self.chain_encoder = _mlp(
                chain_entry_dim, config.chain_embed_dim, config.chain_embed_dim
            )

        # Board representation
        board_input_dim = (
            _NUM_ZONES * config.card_embed_dim + config.global_embed_dim + config.chain_embed_dim
        )
        self.board_mlp = _mlp(board_input_dim, config.board_hidden_dim, config.board_hidden_dim)

        # rnn_type="none" leaves self.rnn=None and head_in_dim=board_hidden_dim,
        # so the state dict stays byte-identical to pre-RNN checkpoints.
        if config.rnn_type == "none":
            self.rnn = None
            head_in_dim = config.board_hidden_dim
        else:
            rnn_cls = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=config.board_hidden_dim,
                hidden_size=config.rnn_hidden_dim,
                num_layers=config.rnn_num_layers,
                batch_first=False,
            )
            head_in_dim = config.rnn_hidden_dim

        # Action encoder
        # Inputs per action: code_emb + desc_passcode_emb (reuses card table)
        # + sysstring_emb (masked when per-card) + ACTION_FEAT_DIM floats.
        self.sysstring_emb = nn.Embedding(SYSSTRING_VOCAB, config.desc_n_embed_dim)
        action_input_dim = (
            embed_dim  # action_codes → card embedding
            + embed_dim  # desc_passcodes → reuses card embedding
            + config.desc_n_embed_dim  # sysstring desc → dedicated table
            + ACTION_FEAT_DIM
        )
        self.action_encoder = _mlp(
            action_input_dim, config.action_embed_dim, config.action_embed_dim
        )

        # Policy head: project board (or RNN output) → action_embed_dim for dot product
        self.board_proj = nn.Linear(head_in_dim, config.action_embed_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @classmethod
    def from_config(cls, config: TrainingConfig) -> YuGiOhNet:
        """Build from config (training time, may load embeddings from file)."""
        text_lookup = None
        if config.card_embeddings:
            text_lookup = TextEmbeddingLookup.from_path(
                config.card_embeddings, config.text_embed_dim
            )
        return cls(config, text_lookup)

    @classmethod
    def from_state_dict(
        cls, config: TrainingConfig, state_dict: dict[str, torch.Tensor]
    ) -> YuGiOhNet:
        """Reconstruct from saved state dict (no disk I/O)."""
        text_prefix = "text_lookup."
        has_text_lookup = any(k.startswith(text_prefix) for k in state_dict)

        text_lookup = None
        if has_text_lookup:
            text_sd = {
                k[len(text_prefix) :]: v for k, v in state_dict.items() if k.startswith(text_prefix)
            }
            text_lookup = TextEmbeddingLookup.from_state_dict_shapes(config.text_embed_dim, text_sd)

        has_rnn_keys = any(k.startswith("rnn.") for k in state_dict)
        if has_rnn_keys != config.is_recurrent:
            raise ValueError(
                f"checkpoint state_dict / config mismatch: "
                f"rnn.* keys present={has_rnn_keys} but "
                f"config.rnn_type={config.rnn_type!r}"
            )

        net = cls(config, text_lookup)
        net.load_state_dict(state_dict)
        return net

    def init_hx(self, batch_size: int, device) -> HxState:
        """Zero hidden state shaped ``(num_layers, batch_size, rnn_hidden_dim)``.

        Returns ``(h, c)`` for LSTM, ``h`` for GRU, ``None`` when no RNN.
        """
        if self.rnn is None:
            return None
        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        if isinstance(self.rnn, nn.LSTM):
            return (h, torch.zeros_like(h))
        return h

    @staticmethod
    def mask_hx(hx: HxState, dones: torch.Tensor) -> HxState:
        """Zero hidden state entries for envs where ``dones[i]`` is True.

        ``dones`` is a 1-D ``(N,)`` tensor; it broadcasts to ``(1, N, 1)``
        against the ``(num_layers, N, hidden)`` hx layout.
        """
        if hx is None:
            return None
        keep = (1.0 - dones.to(dtype=torch.float32)).view(1, -1, 1)
        if isinstance(hx, tuple):
            h, c = hx
            return (h * keep, c * keep)
        return hx * keep

    @staticmethod
    def detach_hx(hx: HxState) -> HxState:
        """Stop gradient flow through hx.

        Used at TBPTT chunk boundaries: each chunk's backward should only
        see the L steps it ran; carrying a grad-tracking hx into the next
        chunk would walk gradients all the way back to t=0.
        """
        if hx is None:
            return None
        if isinstance(hx, tuple):
            return tuple(t.detach() for t in hx)
        return hx.detach()

    @staticmethod
    def slice_hx(
        hx: HxState,
        env_idx: torch.Tensor,
    ) -> HxState:
        """Index hx along the env dimension (dim 1)."""
        if hx is None:
            return None
        if isinstance(hx, tuple):
            return tuple(t[:, env_idx] for t in hx)
        return hx[:, env_idx]

    @staticmethod
    def cat_hx(
        per_env: list[HxState],
        device: torch.device,
    ) -> HxState:
        """Concatenate per-env hx tensors along the env dimension.

        Inverse of ``slice_hx``: takes N per-env hx (each shape
        ``(num_layers, 1, hidden_dim)`` or a tuple thereof) and returns one
        batched hx of shape ``(num_layers, N, hidden_dim)`` on ``device``.
        Used by the actor-learner ingest path to seed the GAE bootstrap.
        """
        sample = per_env[0]
        if sample is None:
            return None
        if isinstance(sample, tuple):
            h = torch.cat([hx[0] for hx in per_env], dim=1).to(device)
            c = torch.cat([hx[1] for hx in per_env], dim=1).to(device)
            return (h, c)
        return torch.cat(per_env, dim=1).to(device)

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
        hx: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        seq_shape: tuple[int, int] | None = None,
        dones: torch.Tensor | None = None,
        obs_chain: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
    ]:
        """Action logits, state value, and the new recurrent hidden state.

        ``seq_shape=None`` is the single-step path used by collection,
        eval, and serving (B == N envs).  ``seq_shape=(T, N)`` activates
        the TBPTT chunk path used by the PPO update — ``dones`` of shape
        ``(T, N)`` is required there to mask hx between micro-steps.
        Single-step / collection callers apply ``mask_hx`` themselves
        between env steps.

        Args:
            obs_cards: (B, 200, 42) uint8 — B is N or T*N depending on path.
            obs_global: (B, 20) uint8
            obs_actions: (B, 32, 28) uint8
            action_mask: (B, 32) int8 — 1=legal, 0=illegal.
            hx: LSTM tuple, GRU tensor, or ``None``.
            obs_chain: (B, MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES) uint8 or ``None``.

        Returns ``(logits (B,32), values (B,), new_hx)``; ``new_hx`` matches
        the structure of ``hx`` (or ``None`` if no RNN).
        """
        # --- Decode observations ---
        card_ids, card_feats = decode_cards(obs_cards)  # (B,200), (B,200,F_card)
        global_feats = decode_global(obs_global)  # (B,F_global)
        # decode_actions returns (codes, desc_passcodes, desc_ns, action_feats);
        # desc_ns is clamped to SYSSTRING_VOCAB-1 for safe embedding lookup.
        action_codes, desc_passcodes, desc_ns, action_feats = decode_actions(obs_actions)

        # --- Card encoding ---
        card_embed = self._embed_codes(card_ids)  # (B, 200, embed_dim)
        card_input = torch.cat([card_embed, card_feats], dim=-1)
        card_enc = self.card_encoder(card_input)  # (B, 200, card_embed_dim)

        # --- Zone pooling ---
        # Extract raw location byte and controller byte for zone assignment
        raw_loc = obs_cards[..., 4].long()  # (B, 200)
        raw_ctrl = obs_cards[..., 7].long()  # (B, 200)

        zone_parts = []
        for ctrl in (0, 1):
            ctrl_mask = raw_ctrl == ctrl  # (B, 200)
            for bit in _ZONE_LOC_BITS:
                # Cards in this zone for this controller
                loc_mask = ((raw_loc & bit) != 0) & ctrl_mask  # (B, 200)
                # Masked mean pooling
                mask_f = loc_mask.float().unsqueeze(-1)  # (B, 200, 1)
                masked_sum = (card_enc * mask_f).sum(dim=1)  # (B, card_embed_dim)
                count = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
                zone_parts.append(masked_sum / count)  # (B, card_embed_dim)

        zone_flat = torch.cat(zone_parts, dim=-1)  # (B, 12*card_embed_dim)

        # --- Global encoding ---
        global_enc = self.global_encoder(global_feats)  # (B, global_embed_dim)

        # --- Pending chain encoding (optional) ---
        if self._chain_embed_dim > 0 and obs_chain is not None:
            chain_codes, chain_desc_passcodes, chain_desc_ns, chain_feats = decode_pending_chain(
                obs_chain
            )

            chain_card_embed = self._embed_codes(chain_codes)
            chain_desc_embed = self._embed_codes(chain_desc_passcodes)

            # desc_n sysstring embedding, masked to 0 when per-card
            # (same pattern as action features — per_card_desc_n scalar
            # is already in chain_feats from decode_pending_chain)
            is_sysstring = chain_desc_passcodes == 0
            sys_embed = self.sysstring_emb(
                chain_desc_ns.clamp(max=self.sysstring_emb.num_embeddings - 1)
            )
            sys_emb_masked = sys_embed * is_sysstring.float().unsqueeze(-1)

            chain_input = torch.cat(
                [chain_card_embed, chain_desc_embed, sys_emb_masked, chain_feats], dim=-1
            )
            chain_enc = self.chain_encoder(chain_input)  # (B, 8, chain_embed_dim)

            # Mean-pool, masking zero-code entries
            mask = (chain_codes != 0).float().unsqueeze(-1)  # (B, 8, 1)
            count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
            chain_pooled = (chain_enc * mask).sum(dim=1) / count  # (B, chain_embed_dim)
        else:
            chain_pooled = (
                torch.zeros(zone_flat.shape[0], self._chain_embed_dim, device=zone_flat.device)
                if self._chain_embed_dim > 0
                else None
            )

        # --- Board representation ---
        if chain_pooled is not None:
            board_input = torch.cat([zone_flat, global_enc, chain_pooled], dim=-1)
        else:
            board_input = torch.cat([zone_flat, global_enc], dim=-1)
        board = self.board_mlp(board_input)  # (B, board_hidden_dim)

        # --- Recurrent layer (optional) ---
        if self.rnn is None:
            head_input = board
            new_hx = None
        elif seq_shape is None:
            step_out, new_hx = self.rnn(board.unsqueeze(0), hx)
            head_input = step_out.squeeze(0)
        else:
            T, N = seq_shape
            assert dones is not None, "TBPTT path requires `dones`"
            seq = board.view(T, N, -1)
            outs = []
            cur_hx = hx
            for t in range(T):
                step_out, cur_hx = self.rnn(seq[t : t + 1], cur_hx)
                outs.append(step_out)
                # Mask AFTER emitting step t so step t still sees pre-done hx.
                cur_hx = self.mask_hx(cur_hx, dones[t])
            head_input = torch.cat(outs, dim=0).reshape(T * N, -1)
            new_hx = cur_hx

        # --- Action encoding ---
        # code_emb: prompt-level card the action targets (e.g. the card to summon).
        # desc_card_emb: card-identity component of the engine effect string id;
        #   reuses the card embedding table (row 0 = sentinel "no card", which is
        #   also what we hit when the desc is a sysstring (passcode == 0)).
        # sys_emb_masked: sysstring component, masked to 0 when the desc is per-card,
        #   so exactly one of (sys_emb, per_card_desc_n_scalar in action_feats) is
        #   non-zero per action.
        act_embed = self._embed_codes(action_codes)  # (B, 32, embed_dim)
        desc_card_embed = self._embed_codes(desc_passcodes)  # (B, 32, embed_dim)
        is_sysstring = desc_passcodes == 0
        sys_emb = self.sysstring_emb(desc_ns)  # (B, 32, desc_n_embed_dim)
        sys_emb_masked = sys_emb * is_sysstring.float().unsqueeze(-1)
        act_input = torch.cat(
            [act_embed, desc_card_embed, sys_emb_masked, action_feats],
            dim=-1,
        )
        act_enc = self.action_encoder(act_input)  # (B, 32, action_embed_dim)

        # --- Policy head: dot product ---
        board_p = self.board_proj(head_input)  # (B, action_embed_dim)
        logits = (act_enc * board_p.unsqueeze(1)).sum(dim=-1)  # (B, 32)

        # Mask illegal actions
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # --- Value head ---
        values = self.value_head(head_input).squeeze(-1)  # (B,)

        return logits, values, new_hx
