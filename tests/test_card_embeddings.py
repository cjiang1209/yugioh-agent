"""Tests for card text embedding support."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import TextEmbeddingLookup, YuGiOhNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings_file(
    codes: list[int],
    dim: int = 384,
    path: str | None = None,
) -> str:
    """Create a synthetic card_text_embeddings.pt file."""
    n = len(codes)
    embeddings = torch.randn(n, dim)
    codes_tensor = torch.tensor(codes, dtype=torch.int64)
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        path = tmp.name
        tmp.close()
    torch.save({
        "embeddings": embeddings,
        "codes": codes_tensor,
        "model_name": "test-model",
    }, path)
    return path


def _make_dummy_obs(batch_size: int = 2):
    """Create minimal dummy observations for a forward pass."""
    obs_cards = torch.zeros(batch_size, 200, 42, dtype=torch.uint8)
    obs_global = torch.zeros(batch_size, 20, dtype=torch.uint8)
    obs_actions = torch.zeros(batch_size, 32, 12, dtype=torch.uint8)
    action_mask = torch.ones(batch_size, 32, dtype=torch.int8)
    # Set at least one legal action
    action_mask[:, 0] = 1
    return obs_cards, obs_global, obs_actions, action_mask


# ---------------------------------------------------------------------------
# TextEmbeddingLookup tests
# ---------------------------------------------------------------------------

class TestTextEmbeddingLookup:
    def test_basic_lookup_shape(self, tmp_path):
        codes = [100, 200, 300]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        input_codes = torch.tensor([100, 200, 300, 999])
        text_embed, embed_idx = lookup(input_codes)

        assert text_embed.shape == (4, 64)
        assert embed_idx.shape == (4,)

    def test_known_codes_get_nonzero_indices(self, tmp_path):
        codes = [10, 20, 30]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        input_codes = torch.tensor([10, 20, 30])
        _, embed_idx = lookup(input_codes)

        # All known codes should map to indices 1, 2, or 3 (not 0)
        assert (embed_idx > 0).all()

    def test_unknown_codes_get_zero_index(self, tmp_path):
        codes = [10, 20, 30]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        input_codes = torch.tensor([999, 0, 5555])
        _, embed_idx = lookup(input_codes)

        assert (embed_idx == 0).all()

    def test_zero_code_maps_to_padding(self, tmp_path):
        codes = [10, 20]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        input_codes = torch.tensor([0])
        text_embed, embed_idx = lookup(input_codes)

        assert embed_idx.item() == 0

    def test_batched_lookup(self, tmp_path):
        codes = [100, 200, 300]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=32)

        input_codes = torch.tensor([[100, 200], [300, 999]])
        text_embed, embed_idx = lookup(input_codes)

        assert text_embed.shape == (2, 2, 32)
        assert embed_idx.shape == (2, 2)

    def test_num_cards_property(self, tmp_path):
        codes = [10, 20, 30, 40]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        assert lookup.num_cards == 4

    def test_buffer_moves_with_device(self, tmp_path):
        codes = [10, 20]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        # Move to CPU explicitly (trivial but verifies buffer registration)
        lookup = lookup.to("cpu")
        assert lookup._sorted_codes.device.type == "cpu"

        input_codes = torch.tensor([10, 20])
        text_embed, embed_idx = lookup(input_codes)
        assert text_embed.device.type == "cpu"

    def test_projection_is_trainable(self, tmp_path):
        codes = [10, 20]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        trainable = [n for n, p in lookup.named_parameters() if p.requires_grad]
        assert "_proj.weight" in trainable
        assert "_proj.bias" in trainable

    def test_frozen_embed_not_trainable(self, tmp_path):
        codes = [10, 20]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))
        lookup = TextEmbeddingLookup(path, text_embed_dim=64)

        frozen_params = [
            n for n, p in lookup.named_parameters()
            if "_frozen_embed" in n and p.requires_grad
        ]
        assert len(frozen_params) == 0


# ---------------------------------------------------------------------------
# YuGiOhNet symbolic mode (default, no text embeddings) — regression test
# ---------------------------------------------------------------------------

class TestSymbolicMode:
    def test_forward_pass_no_text_embeddings(self):
        config = TrainingConfig()
        assert config.card_embeddings_path == ""

        net = YuGiOhNet(config)
        obs_cards, obs_global, obs_actions, action_mask = _make_dummy_obs(batch_size=2)
        logits, values = net(obs_cards, obs_global, obs_actions, action_mask)

        assert logits.shape == (2, 32)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# YuGiOhNet semantic mode (text embeddings enabled)
# ---------------------------------------------------------------------------

class TestSemanticMode:
    def test_forward_pass_with_text_embeddings(self, tmp_path):
        codes = list(range(1, 101))  # 100 synthetic cards
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))

        config = TrainingConfig(
            card_embeddings_path=path,
            text_embed_dim=64,
            learned_embed_dim=8,
        )
        net = YuGiOhNet(config)
        obs_cards, obs_global, obs_actions, action_mask = _make_dummy_obs(batch_size=2)
        logits, values = net(obs_cards, obs_global, obs_actions, action_mask)

        assert logits.shape == (2, 32)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()

    def test_text_lookup_is_present(self, tmp_path):
        codes = [10, 20]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))

        config = TrainingConfig(card_embeddings_path=path)
        net = YuGiOhNet(config)

        assert net.text_lookup is not None
        assert net._use_text_embeddings is True

    def test_symbolic_mode_has_no_text_lookup(self):
        config = TrainingConfig()
        net = YuGiOhNet(config)

        assert net.text_lookup is None
        assert net._use_text_embeddings is False

    def test_embed_dim_matches_config(self, tmp_path):
        codes = [10, 20, 30]
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))

        config = TrainingConfig(
            card_embeddings_path=path,
            text_embed_dim=48,
            learned_embed_dim=12,
        )
        net = YuGiOhNet(config)

        # Card encoder input should be text_embed_dim + learned_embed_dim + CARD_FEAT_DIM
        from yugioh_rl.features import CARD_FEAT_DIM
        expected_input_dim = 48 + 12 + CARD_FEAT_DIM
        assert net.card_encoder[0].in_features == expected_input_dim

    def test_backward_pass(self, tmp_path):
        codes = list(range(1, 51))
        path = _make_embeddings_file(codes, dim=384, path=str(tmp_path / "emb.pt"))

        config = TrainingConfig(card_embeddings_path=path, text_embed_dim=32, learned_embed_dim=8)
        net = YuGiOhNet(config)

        obs_cards, obs_global, obs_actions, action_mask = _make_dummy_obs(batch_size=4)
        logits, values = net(obs_cards, obs_global, obs_actions, action_mask)

        loss = values.mean() + logits[:, 0].mean()
        loss.backward()

        # Projection should have gradients
        assert net.text_lookup._proj.weight.grad is not None
        # Learned embedding should have gradients
        assert net.card_embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# build_card_embeddings.py end-to-end test
# ---------------------------------------------------------------------------

class TestBuildEmbeddingsScript:
    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a mock cards.cdb with a few rows."""
        db_path = tmp_path / "cards.cdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE texts (id INTEGER PRIMARY KEY, name TEXT, desc TEXT)")
        conn.executemany(
            "INSERT INTO texts (id, name, desc) VALUES (?, ?, ?)",
            [
                (12345, "Blue-Eyes White Dragon", "This legendary dragon is a powerful engine of destruction."),
                (67890, "Dark Magician", "The ultimate wizard in terms of attack and defense."),
                (11111, "Empty Card", ""),
            ],
        )
        conn.commit()
        conn.close()
        return str(db_path)

    def test_load_card_texts(self, mock_db):
        from cli.build_card_embeddings import load_card_texts

        codes, descriptions = load_card_texts(mock_db)

        assert len(codes) == 3
        assert len(descriptions) == 3
        assert 12345 in codes
        assert 67890 in codes
        assert 11111 in codes
        # Empty card should have empty string
        idx = codes.index(11111)
        assert descriptions[idx] == ""

    def test_build_output_structure(self, mock_db, tmp_path):
        """End-to-end test: requires sentence-transformers."""
        st = pytest.importorskip("sentence_transformers")

        import sys
        sys.argv = [
            "build_card_embeddings.py",
            "--db", mock_db,
            "--output", str(tmp_path / "test_emb.pt"),
        ]

        from cli.build_card_embeddings import main
        main()

        output_path = tmp_path / "test_emb.pt"
        assert output_path.exists()

        data = torch.load(str(output_path), weights_only=True)
        assert "embeddings" in data
        assert "codes" in data
        assert "model_name" in data

        assert data["embeddings"].shape[0] == 3
        assert data["embeddings"].shape[1] == 384  # all-MiniLM-L6-v2 output dim
        assert data["codes"].shape == (3,)

        # Empty card should have zero embedding
        codes_list = data["codes"].tolist()
        empty_idx = codes_list.index(11111)
        assert torch.allclose(data["embeddings"][empty_idx], torch.zeros(384))
