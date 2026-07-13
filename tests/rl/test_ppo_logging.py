"""PPOTrainer emits the right logging events via an injected sink (no real I/O)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yugioh_rl.metrics_logging import CheckpointEvent, MultiSink


class _FakeSink:
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)

    def close(self):
        pass


def _make_trainer_stub(config, sinks):
    """Build a PPOTrainer-like object with just what _save_checkpoint reads."""
    from yugioh_rl.ppo import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.config = config
    trainer.network = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.Adam(trainer.network.parameters())
    trainer._episode_rewards = []
    trainer._episode_lengths = []
    trainer._episode_wins = []
    trainer._deck_wins = {}
    trainer._opponent_pool = None
    trainer._sinks = sinks
    return trainer


def test_save_checkpoint_emits_registration_event(tmp_path):
    from yugioh_rl.config import TrainingConfig

    fake = _FakeSink()
    config = TrainingConfig(
        deck_paths=["assets/decks/blue_eyes.ydk"],
        save_dir=str(tmp_path),
        device="cpu",
        total_timesteps=1,
        num_envs=1,
        rollout_steps=1,
        seed=42,
        log_to=["tensorboard"],
    )
    trainer = _make_trainer_stub(config, MultiSink([fake]))

    trainer._save_checkpoint(update=100, global_step=2048)

    ckpt_events = [e for e in fake.events if isinstance(e, CheckpointEvent)]
    assert len(ckpt_events) == 1
    ev = ckpt_events[0]
    assert ev.scalars == {}  # registration, not measurement
    assert ev.ref.update == 100
    assert ev.ref.global_step == 2048
    assert ev.ref.path == Path(config.save_dir) / "checkpoint_100.pt"
    assert ev.ref.params["seed"] == "42"
    assert "feature_signature" in ev.ref.params
    assert ev.ref.tags == {}  # no user tags; run provenance is the source-run link
    assert (Path(config.save_dir) / "checkpoint_100.pt").exists()  # Path 3: disk write stays
