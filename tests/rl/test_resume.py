"""Tests for --resume training resumption."""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
)
from yugioh_rl.config import TrainingConfig
from yugioh_rl.network import YuGiOhNet
from yugioh_rl.ppo import PPOTrainer


def _make_checkpoint(
    path: str,
    config: TrainingConfig | None = None,
    run_backward: bool = False,
    update: int = 10,
    global_step: int = 1000,
    episode_rewards: list[float] | None = None,
    episode_lengths: list[int] | None = None,
    episode_wins: list[float] | None = None,
    deck_wins: dict[int, list[float]] | None = None,
    omit_lists: bool = False,
) -> YuGiOhNet:
    """Create a checkpoint and return the network used to build it."""
    if config is None:
        config = TrainingConfig()
    net = YuGiOhNet.from_config(config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    if run_backward:
        cards = torch.zeros(1, MAX_CARDS, CARD_FEATURES, dtype=torch.uint8)
        glob = torch.zeros(1, GLOBAL_FEATURES, dtype=torch.uint8)
        actions = torch.zeros(1, MAX_ACTIONS, ACTION_FEATURES, dtype=torch.uint8)
        mask = torch.ones(1, MAX_ACTIONS, dtype=torch.int8)
        for _ in range(3):
            logits, value, _ = net(cards, glob, actions, mask)
            loss = -logits.mean() + value.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    data = {
        "update": update,
        "global_step": global_step,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "episode_rewards": episode_rewards or [0.5, -0.5, 1.0],
    }
    if not omit_lists:
        data["episode_lengths"] = episode_lengths or [10, 20, 30]
        data["episode_wins"] = episode_wins or [1.0, 0.0, 1.0]
    if deck_wins is not None:
        data["deck_wins"] = deck_wins

    torch.save(data, path)
    return net


def test_resume_restores_counters(tmp_path):
    """_resume_update and _resume_global_step should match the checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, update=25, global_step=5000)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._resume_update == 25
    assert trainer._resume_global_step == 5000


def test_resume_restores_weights(tmp_path):
    """Model weights should match the checkpoint exactly."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    config = TrainingConfig(save_dir=str(tmp_path), num_envs=1)
    net = _make_checkpoint(ckpt_path, config)
    ref_param = next(iter(net.parameters())).detach().clone()

    resume_config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(resume_config)
    loaded_param = next(iter(trainer.network.parameters())).detach()

    assert torch.allclose(ref_param, loaded_param), "Weights should match checkpoint"


def test_resume_restores_optimizer_state(tmp_path):
    """Optimizer state should be populated after resume."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, run_backward=True)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert len(trainer.optimizer.state) > 0, "Optimizer state should be populated"


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_resume_restores_recurrent_weights(tmp_path, rnn_type):
    """Resume must round-trip the RNN module, not just the feed-forward
    layers.  Probes ``rnn.weight_ih_l0`` directly — ``next(iter(net
    .parameters()))`` returns ``card_embedding.weight`` regardless of
    rnn_type, so it would silently pass even if a regression dropped or
    re-initialised the recurrent layer.
    """
    import dataclasses

    ckpt_path = str(tmp_path / "ckpt.pt")
    config = TrainingConfig(
        save_dir=str(tmp_path), num_envs=1,
        rnn_type=rnn_type, rnn_hidden_dim=64, rnn_num_layers=1,
        bptt_chunk_len=8, rollout_steps=8, minibatch_size=8,
    )
    net = _make_checkpoint(ckpt_path, config, run_backward=True)
    ref_param = net.state_dict()["rnn.weight_ih_l0"].detach().clone()

    trainer = PPOTrainer(dataclasses.replace(config, resume_checkpoint=ckpt_path))
    loaded_param = trainer.network.state_dict()["rnn.weight_ih_l0"].detach()

    assert torch.allclose(ref_param, loaded_param)
    assert len(trainer.optimizer.state) > 0


def test_resume_restores_episode_tracking(tmp_path):
    """Episode tracking lists should be restored from checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    rewards = [1.0, -1.0, 0.5]
    lengths = [15, 25, 35]
    wins = [1.0, 0.0, 1.0]
    _make_checkpoint(
        ckpt_path,
        episode_rewards=rewards,
        episode_lengths=lengths,
        episode_wins=wins,
    )

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._episode_rewards == rewards
    assert trainer._episode_lengths == lengths
    assert trainer._episode_wins == wins


def test_resume_backward_compat_missing_lists(tmp_path):
    """Old checkpoints without episode_lengths/wins should resume with empty lists."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path, omit_lists=True)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._episode_rewards == [0.5, -0.5, 1.0]  # present in checkpoint
    assert trainer._episode_lengths == []  # missing → empty fallback
    assert trainer._episode_wins == []  # missing → empty fallback


def test_resume_deck_paths_mismatch_rejected(tmp_path):
    """Direct (non-CLI) callers that resume without first merging deck_paths
    from the checkpoint should get a loud error rather than silently
    misattributing index-keyed _deck_wins to the wrong deck names."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    ckpt_config = TrainingConfig(
        deck_paths=["assets/decks/blue_eyes.ydk", "assets/decks/dark_magician.ydk"],
    )
    _make_checkpoint(ckpt_path, config=ckpt_config)

    # Caller forgets to restore deck_paths; dataclass default is single-deck.
    resume_config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    with pytest.raises(ValueError, match="deck_paths mismatch"):
        PPOTrainer(resume_config)


def test_resume_arch_mismatch_rejected(tmp_path):
    """Architecture mismatch should raise ValueError."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
        card_embed_dim=128,
    )
    with pytest.raises(ValueError, match="Architecture mismatch"):
        PPOTrainer(config)


def test_resume_infers_save_dir(tmp_path):
    """save_dir should be the checkpoint's parent directory."""
    run_dir = tmp_path / "runs" / "20260101_120000_seed42"
    run_dir.mkdir(parents=True)
    ckpt_path = str(run_dir / "checkpoint_10.pt")
    _make_checkpoint(ckpt_path)

    config = TrainingConfig(
        save_dir=str(run_dir),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer.config.save_dir == str(run_dir)


def test_resume_lr_override(tmp_path):
    """Optimizer LR should use the CLI value, not the checkpoint's stale LR."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    old_config = TrainingConfig(learning_rate=1e-3)
    _make_checkpoint(ckpt_path, config=old_config, run_backward=True)

    new_lr = 5e-5
    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
        learning_rate=new_lr,
    )
    trainer = PPOTrainer(config)

    for pg in trainer.optimizer.param_groups:
        assert pg["lr"] == new_lr, f"Expected LR {new_lr}, got {pg['lr']}"


def test_resume_early_return_when_training_complete(tmp_path, caplog):
    """train() should return immediately if resume update >= num_updates."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    # Checkpoint at update=10, global_step=20480 (10 * 256 * 8)
    _make_checkpoint(ckpt_path, update=10, global_step=20480)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=8,
        rollout_steps=256,
        # total_timesteps that yields exactly 10 updates → already done
        total_timesteps=256 * 8 * 10,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    import logging
    with caplog.at_level(logging.WARNING):
        trainer.train()

    assert "training already complete" in caplog.text


def test_resume_restores_deck_wins(tmp_path):
    """Per-deck win history should be restored from checkpoint."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    saved_deck_wins = {0: [1.0, 0.0, 1.0], 1: [0.0, 0.0, 1.0]}
    _make_checkpoint(ckpt_path, deck_wins=saved_deck_wins)

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._deck_wins == saved_deck_wins


def test_resume_backward_compat_missing_deck_wins(tmp_path):
    """Old checkpoints without deck_wins should resume with empty dict."""
    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path)  # no deck_wins

    config = TrainingConfig(
        save_dir=str(tmp_path),
        num_envs=1,
        resume_checkpoint=ckpt_path,
    )
    trainer = PPOTrainer(config)

    assert trainer._deck_wins == {}


# ---------------------------------------------------------------------------
# CLI-level resume-config merge (`_build_resume_config` + snapshot writer)
# ---------------------------------------------------------------------------
#
# These tests exercise the CLI-side flow that loads ckpt["config"], applies
# allowlisted overrides, and writes the `config.json` snapshot.  The deck_paths
# restoration previously lived inside PPOTrainer; it now lives here.


def test_resume_restores_deck_paths_via_cli(tmp_path, monkeypatch):
    """On --resume, the CLI merges deck_paths from ckpt["config"] so the
    index-keyed _deck_wins map stays aligned with the checkpoint."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    original_paths = ["assets/decks/blue_eyes.ydk", "assets/decks/dark_magician.ydk"]
    ckpt_config = TrainingConfig(deck_paths=original_paths)
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    config = _build_resume_config(args, str(tmp_path))

    assert config.deck_paths == original_paths


def test_resume_reads_config_from_checkpoint(tmp_path, monkeypatch):
    """Without CLI overrides, resume config should match the ckpt values."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    ckpt_config = TrainingConfig(
        total_timesteps=500_000,
        learning_rate=1e-3,
        num_envs=4,
        opponent="random",
        seed=123,
    )
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    config = _build_resume_config(args, str(tmp_path))

    assert config.total_timesteps == 500_000
    assert config.learning_rate == 1e-3
    assert config.num_envs == 4
    assert config.opponent == "random"
    assert config.seed == 123
    assert config.resume_checkpoint == ckpt_path
    assert config.save_dir == str(tmp_path)
    assert config.init_checkpoint == ""
    assert config.init_optimizer is False


def test_resume_cli_override_allowlist(tmp_path, monkeypatch):
    """An allowlisted flag passed on the CLI should win over the ckpt value."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    ckpt_config = TrainingConfig(total_timesteps=500_000, learning_rate=1e-3)
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", [
        "cli.train",
        "--resume", ckpt_path,
        "--total-timesteps", "2000000",
        "--learning-rate", "5e-5",
    ])
    args = parse_args()
    config = _build_resume_config(args, str(tmp_path))

    assert config.total_timesteps == 2_000_000
    assert config.learning_rate == 5e-5


def test_resume_non_allowlist_override_rejected(tmp_path, monkeypatch, capsys):
    """Passing a non-allowlisted flag with --resume should _fatal and name
    the offending flag before any torch.load runs."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    _make_checkpoint(ckpt_path)

    monkeypatch.setattr(sys, "argv", [
        "cli.train", "--resume", ckpt_path, "--card-embed-dim", "128",
    ])
    args = parse_args()
    with pytest.raises(SystemExit):
        _build_resume_config(args, str(tmp_path))
    err = capsys.readouterr().err
    assert "--card-embed-dim" in err
    assert "cannot be overridden" in err


def test_resume_validates_effective_config(tmp_path, monkeypatch, capsys):
    """A ckpt-stored deck path that no longer exists should fail post-merge
    validation rather than silently training on a nonexistent file."""
    from cli.train import (
        _build_resume_config,
        parse_args,
        validate_effective_config,
    )

    ckpt_path = str(tmp_path / "ckpt.pt")
    missing_deck = str(tmp_path / "nonexistent.ydk")
    ckpt_config = TrainingConfig(deck_paths=[missing_deck])
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    config = _build_resume_config(args, str(tmp_path))

    with pytest.raises(SystemExit):
        validate_effective_config(config)
    err = capsys.readouterr().err
    assert "not found" in err
    assert missing_deck in err


def test_resume_legacy_missing_field_silently_backfilled(tmp_path, monkeypatch):
    """Pre-existing ckpt whose __dict__ is missing a field added later should
    resume cleanly, with the missing field back-filled to its dataclass default.

    Replaces the old strict schema-drift check on additive fields — see
    yugioh_rl/config.normalize_legacy_config.
    """
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    ckpt_config = TrainingConfig()
    del ckpt_config.__dict__["agent_player"]
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    cfg = _build_resume_config(args, str(tmp_path))
    assert cfg.agent_player == "random"


def test_resume_schema_drift_extra_field_errors(tmp_path, monkeypatch, capsys):
    """Ckpt_config carrying an unknown attribute should _fatal."""
    from cli.train import _build_resume_config, parse_args

    ckpt_path = str(tmp_path / "ckpt.pt")
    ckpt_config = TrainingConfig()
    ckpt_config.__dict__["some_future_field"] = "x"
    _make_checkpoint(ckpt_path, config=ckpt_config)

    monkeypatch.setattr(sys, "argv", ["cli.train", "--resume", ckpt_path])
    args = parse_args()
    with pytest.raises(SystemExit):
        _build_resume_config(args, str(tmp_path))
    err = capsys.readouterr().err
    assert "schema" in err
    assert "some_future_field" in err


def test_write_config_snapshot_creates_timestamped_file_and_symlink(tmp_path):
    """_write_config_snapshot should write config_{timestamp}.json and
    repoint config.json as a symlink to it."""
    from cli.train import _write_config_snapshot

    cfg = TrainingConfig(save_dir=str(tmp_path))
    snapshot_path = _write_config_snapshot(cfg)

    assert snapshot_path.exists()
    assert snapshot_path.name.startswith("config_")
    assert snapshot_path.name.endswith(".json")

    symlink = tmp_path / "config.json"
    assert symlink.is_symlink()
    assert symlink.resolve() == snapshot_path.resolve()


def test_write_config_snapshot_replaces_preexisting_plain_file(tmp_path):
    """If the run directory already contains a plain (non-symlink) config.json
    from an older codebase version, _write_config_snapshot should unlink it
    and replace with a fresh symlink. The original plain file's contents are
    not preserved."""
    from cli.train import _write_config_snapshot

    legacy = tmp_path / "config.json"
    legacy.write_text('{"legacy": true}')
    assert not legacy.is_symlink()

    cfg = TrainingConfig(save_dir=str(tmp_path))
    snapshot_path = _write_config_snapshot(cfg)

    assert legacy.is_symlink()
    assert legacy.resolve() == snapshot_path.resolve()
