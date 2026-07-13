"""run_sweep emits one CheckpointEvent per (checkpoint, opponent) via the sink."""

from cli.eval_sweep import Manifest, run_sweep

from yugioh_rl.metrics_logging import CheckpointEvent


class _FakeSink:
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)

    def close(self):
        pass


class _EvalResultStub:
    def __init__(self, win_rate, wins, episodes, per_deck_wins):
        self.opponent_label = "greedy"
        self.win_rate = win_rate
        self.wins = wins
        self.episodes = episodes
        self.per_deck_wins = per_deck_wins


def _fake_evaluate(**kwargs):
    return [_EvalResultStub(win_rate=1.0, wins=1, episodes=1, per_deck_wins={0: [1.0]})]


def _fake_load(path, map_location=None, weights_only=None):
    return {"global_step": 4096, "config": None}


def test_run_sweep_emits_checkpoint_event_per_pair(tmp_path):
    ckpt = tmp_path / "checkpoint_200.pt"
    ckpt.write_bytes(b"weights")
    fake = _FakeSink()
    manifest = Manifest.load(tmp_path / "manifest.json")

    summary = run_sweep(
        checkpoints=[ckpt],
        opponents=["greedy"],
        deck_pool=[{}],
        deck_paths=["assets/decks/starter.ydk"],
        manifest=manifest,
        sink=fake,
        num_episodes=1,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=_fake_evaluate,
        load_fn=_fake_load,
    )

    assert summary["ok"] == 1
    events = [e for e in fake.events if isinstance(e, CheckpointEvent)]
    assert len(events) == 1
    ev = events[0]
    assert ev.ref.update == 200
    assert ev.ref.global_step == 4096
    assert ev.scalars["win_rate_vs_greedy"] == 1.0
    assert ev.scalars["win_rate_vs_greedy_deck_starter"] == 1.0
