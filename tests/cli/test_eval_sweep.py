from pathlib import Path

import pytest
from cli.eval_sweep import (
    Manifest,
    SweepError,
    _match_opponent_dir,
    _step_matched_decks,
    checkpoint_update,
    derive_deck_paths,
    discover_checkpoints,
    parse_args,
    run_sweep,
)

from yugioh_rl.eval import EvalResult


def _touch(p: Path):
    p.write_bytes(b"")


def test_checkpoint_update_parses_filename():
    assert checkpoint_update(Path("checkpoint_100.pt")) == 100
    assert checkpoint_update(Path("/a/b/checkpoint_2441.pt")) == 2441


def test_discover_sorts_numerically_and_excludes_symlink(tmp_path):
    for n in (100, 200, 1000, 300):  # deliberately unsorted, 1000 > 300 lexically
        _touch(tmp_path / f"checkpoint_{n}.pt")
    # non-numeric name must be excluded (mimics checkpoint_latest.pt)
    _touch(tmp_path / "checkpoint_latest.pt")
    got = [checkpoint_update(p) for p in discover_checkpoints(str(tmp_path), stride=1)]
    assert got == [100, 200, 300, 1000]


def test_discover_stride(tmp_path):
    for n in range(100, 1100, 100):  # 100,200,...,1000  (10 checkpoints)
        _touch(tmp_path / f"checkpoint_{n}.pt")
    got = [checkpoint_update(p) for p in discover_checkpoints(str(tmp_path), stride=5)]
    assert got == [100, 600]  # indices 0 and 5


def test_manifest_absent_file_is_empty(tmp_path):
    m = Manifest.load(tmp_path / "manifest.json")
    assert not m.has(100, "random")
    assert m.get(100, "random") is None


def test_manifest_record_flushes_and_reloads(tmp_path):
    p = tmp_path / "manifest.json"
    m = Manifest.load(p)
    row = {"win_rate": 0.5, "wins": 5, "episodes": 10, "per_deck": {}}
    m.record(100, "random", row)
    assert m.has(100, "random")
    assert m.get(100, "random") == row
    # flushed immediately: a fresh load sees it (valid resume point)
    m2 = Manifest.load(p)
    assert m2.has(100, "random")
    assert m2.get(100, "random") == row


def test_manifest_pair_identity_is_update_and_label(tmp_path):
    m = Manifest.load(tmp_path / "manifest.json")
    m.record(100, "random", {"win_rate": 1.0, "wins": 1, "episodes": 1, "per_deck": {}})
    assert m.has(100, "random")
    assert not m.has(100, "greedy")  # different label
    assert not m.has(200, "random")  # different update


class _Cfg:
    def __init__(self, deck_paths):
        self.deck_paths = deck_paths


def test_derive_uses_override_when_given():
    got = derive_deck_paths(
        [Path("checkpoint_100.pt")], override=["x.ydk"], load_fn=lambda p, **k: {}
    )
    assert got == ["x.ydk"]


def test_derive_from_first_checkpoint_config_object():
    def fake_load(p, **k):
        return {"config": _Cfg(["a.ydk", "b.ydk"])}

    got = derive_deck_paths([Path("checkpoint_100.pt")], override=None, load_fn=fake_load)
    assert got == ["a.ydk", "b.ydk"]


def test_derive_from_config_dict():
    def fake_load(p, **k):
        return {"config": {"deck_paths": ["c.ydk"]}}

    got = derive_deck_paths([Path("checkpoint_100.pt")], override=None, load_fn=fake_load)
    assert got == ["c.ydk"]


def test_derive_skips_unreadable_then_succeeds():
    calls = []

    def fake_load(p, **k):
        calls.append(p)
        if "100" in str(p):
            raise RuntimeError("corrupt")
        return {"config": _Cfg(["d.ydk"])}

    got = derive_deck_paths(
        [Path("checkpoint_100.pt"), Path("checkpoint_200.pt")],
        override=None,
        load_fn=fake_load,
    )
    assert got == ["d.ydk"]
    assert len(calls) == 2  # first failed, second succeeded


def test_derive_raises_when_unresolvable():
    def fake_load(p, **k):
        raise RuntimeError("corrupt")

    with pytest.raises(SweepError):
        derive_deck_paths([Path("checkpoint_100.pt")], override=None, load_fn=fake_load)


def test_match_opponent_dir_pairs_by_update_and_ignores_unmatched():
    run = [Path(f"run/checkpoint_{n}.pt") for n in (100, 200, 300)]
    opp = [Path(f"opp/checkpoint_{n}.pt") for n in (200, 300, 400)]
    matched = _match_opponent_dir(run, opp)
    # matched on N in {200, 300}; run-100 and opp-400 ignored. Run-dir order kept.
    assert [(checkpoint_update(r), checkpoint_update(o)) for r, o in matched] == [
        (200, 200),
        (300, 300),
    ]
    assert matched[0] == (Path("run/checkpoint_200.pt"), Path("opp/checkpoint_200.pt"))


def test_match_opponent_dir_disjoint_yields_empty():
    run = [Path("run/checkpoint_100.pt")]
    opp = [Path("opp/checkpoint_200.pt")]
    assert _match_opponent_dir(run, opp) == []  # main turns this into SweepError/rc 2


def test_step_matched_decks_override_wins():
    got = _step_matched_decks(
        [Path("run/checkpoint_1.pt")],
        [Path("base/checkpoint_1.pt")],
        override=["x.ydk"],
        load_fn=lambda p, **k: {},
    )
    assert got == ["x.ydk"]


def test_step_matched_decks_intersects_configs():
    def fake_load(p, **k):
        decks = ["a.ydk", "b.ydk", "c.ydk"] if "run" in str(p) else ["b.ydk", "c.ydk", "d.ydk"]
        return {"config": _Cfg(decks)}

    got = _step_matched_decks(
        [Path("run/checkpoint_1.pt")],
        [Path("base/checkpoint_1.pt")],
        override=None,
        load_fn=fake_load,
    )
    assert got == ["b.ydk", "c.ydk"]  # intersection, run-dir order preserved


def test_step_matched_decks_empty_intersection_raises():
    def fake_load(p, **k):
        return {"config": _Cfg(["a.ydk"] if "run" in str(p) else ["z.ydk"])}

    with pytest.raises(SweepError):
        _step_matched_decks(
            [Path("run/checkpoint_1.pt")],
            [Path("base/checkpoint_1.pt")],
            override=None,
            load_fn=fake_load,
        )


class _FakeSink:
    """Records emitted events and exposes a ``scalars`` view identical to what a
    real TensorBoardSink would write (unprefixed, at the event's global_step),
    so these tests still assert on byte-identical TB keys."""

    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)

    def close(self):
        pass

    @property
    def scalars(self):
        out = []
        for ev in self.events:
            for key, value in ev.scalars.items():
                out.append((key, value, ev.ref.global_step))
        return out


def _mk_ckpts(tmp_path, updates):
    for n in updates:
        (tmp_path / f"checkpoint_{n}.pt").write_bytes(b"")
    from cli.eval_sweep import discover_checkpoints

    return discover_checkpoints(str(tmp_path), stride=1)


def test_run_sweep_evals_and_records(tmp_path):
    ckpts = _mk_ckpts(tmp_path, [100, 200])
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")

    def fake_eval(**kwargs):
        return [EvalResult("random", 10, 6)]

    def fake_load(p, **k):
        return {"global_step": checkpoint_update(p)}

    summary = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[{"main": [1]}],
        deck_paths=["deck.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=fake_eval,
        load_fn=fake_load,
    )
    assert summary["ok"] == 2 and summary["failed"] == 0 and summary["skipped"] == 0
    # both checkpoints recorded
    assert manifest.has(100, "random") and manifest.has(200, "random")
    # TB got win_rate at each step (global_step == update in this stub)
    assert ("win_rate/random/overall", 0.6, 100) in sink.scalars
    assert ("win_rate/random/overall", 0.6, 200) in sink.scalars


def test_run_sweep_skips_recorded_and_replays_to_tb(tmp_path):
    ckpts = _mk_ckpts(tmp_path, [100])
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")
    manifest.record(100, "random", {"win_rate": 0.9, "wins": 9, "episodes": 10, "per_deck": {}})

    def boom(**kwargs):  # must NOT be called for a recorded pair
        raise AssertionError("evaluate called for already-recorded pair")

    def fake_load_should_not_be_called(p, **k):
        raise AssertionError("load_fn should not be called on the skip/replay path")

    summary = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[],
        deck_paths=["d.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=boom,
        load_fn=fake_load_should_not_be_called,
    )
    assert summary["skipped"] == 1 and summary["ok"] == 0
    # replayed the stored win_rate to TB using fallback global_step (= update = 100)
    assert ("win_rate/random/overall", 0.9, 100) in sink.scalars


def test_run_sweep_failure_is_skipped_not_recorded(tmp_path):
    ckpts = _mk_ckpts(tmp_path, [100, 200])
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")

    def flaky_eval(**kwargs):
        # fail for checkpoint 100, succeed for 200
        if "checkpoint_100" in kwargs["agent_spec"]:
            raise RuntimeError("bridge 500")
        return [EvalResult("random", 10, 5)]

    def fake_load(p, **k):
        return {"global_step": checkpoint_update(p)}

    summary = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[],
        deck_paths=["d.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=flaky_eval,
        load_fn=fake_load,
    )
    assert summary["ok"] == 1 and summary["failed"] == 1
    assert (100, "random") in summary["failures"]
    assert not manifest.has(100, "random")  # failure NOT recorded → retried later
    assert manifest.has(200, "random")


def test_run_sweep_force_reevaluates(tmp_path):
    ckpts = _mk_ckpts(tmp_path, [100])
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")
    manifest.record(100, "random", {"win_rate": 0.1, "wins": 1, "episodes": 10, "per_deck": {}})

    def fake_eval(**kwargs):
        return [EvalResult("random", 10, 8)]

    def fake_load(p, **k):
        return {"global_step": 100}

    summary = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[],
        deck_paths=["d.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=True,
        evaluate_fn=fake_eval,
        load_fn=fake_load,
    )
    assert summary["ok"] == 1 and summary["skipped"] == 0
    assert manifest.get(100, "random")["win_rate"] == 0.8  # overwritten


def test_run_sweep_records_global_step_and_replays_it(tmp_path):
    """Record a result with global_step, then replay it without reloading the checkpoint."""
    ckpts = _mk_ckpts(tmp_path, [100])
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")

    def fake_eval(**kwargs):
        return [EvalResult("random", 10, 6)]

    def fake_load(p, **k):
        # First call: record phase returns global_step=12345
        return {"global_step": 12345}

    # First run: evaluate and record with global_step
    summary1 = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[{"main": [1]}],
        deck_paths=["deck.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=fake_eval,
        load_fn=fake_load,
    )
    assert summary1["ok"] == 1 and summary1["skipped"] == 0
    # Verify global_step was recorded
    recorded_row = manifest.get(100, "random")
    assert recorded_row["global_step"] == 12345

    # Second run: replay the result (with a load_fn that would fail if called)
    sink2 = _FakeSink()

    def boom_load(p, **k):
        raise AssertionError("load_fn should NOT be called on the skip/replay path")

    summary2 = run_sweep(
        pairs=[(c, f"model:{c}", [("random", "random")]) for c in ckpts],
        deck_pool=[{"main": [1]}],
        deck_paths=["deck.ydk"],
        manifest=manifest,
        sink=sink2,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=fake_eval,
        load_fn=boom_load,  # This should NOT be called
    )
    assert summary2["skipped"] == 1 and summary2["ok"] == 0
    # Verify it replayed to TB at the recorded global_step (12345), not at update (100)
    assert ("win_rate/random/overall", 0.6, 12345) in sink2.scalars


def test_run_sweep_cross_play_uses_provided_label(tmp_path):
    """A step-matched-evaluation pair evaluates run-ckpt vs an opponent model: spec,
    labelled by the caller-supplied (constant) series label; event ref is the run ckpt."""
    run_ckpt = tmp_path / "symbolic" / "checkpoint_200.pt"
    opp_ckpt = tmp_path / "opponent" / "checkpoint_200.pt"
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval" / "manifest.json")

    def fake_eval(**kwargs):
        assert kwargs["agent_spec"] == f"model:{run_ckpt}"
        assert kwargs["opponent_specs"] == [f"model:{opp_ckpt}"]
        return [EvalResult("ignored", 10, 7)]

    def fake_load(p, **k):
        return {"global_step": 2048}

    summary = run_sweep(
        pairs=[(run_ckpt, f"model:{run_ckpt}", [(f"model:{opp_ckpt}", "model_opponent")])],
        deck_pool=[{"main": [1]}],
        deck_paths=["deck.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=fake_eval,
        load_fn=fake_load,
    )
    assert summary["ok"] == 1
    # Metric keyed by the provided series label; event ref is the run checkpoint.
    assert ("win_rate/model_opponent/overall", 0.7, 2048) in sink.scalars
    assert sink.events[0].ref.path == run_ckpt
    assert manifest.has(200, "model_opponent")


def test_run_sweep_cross_play_constant_label_is_one_curve(tmp_path):
    """Across matched N, one constant series label => a single metric tag with a
    point per checkpoint (a curve over global_step), not a distinct tag per N."""
    run_100 = tmp_path / "run" / "checkpoint_100.pt"
    run_200 = tmp_path / "run" / "checkpoint_200.pt"
    opp_100 = tmp_path / "opp" / "checkpoint_100.pt"
    opp_200 = tmp_path / "opp" / "checkpoint_200.pt"
    sink = _FakeSink()
    manifest = Manifest.load(tmp_path / "logs" / "eval_vs_opp" / "manifest.json")

    summary = run_sweep(
        pairs=[
            (run_100, f"model:{run_100}", [(f"model:{opp_100}", "model_opp")]),
            (run_200, f"model:{run_200}", [(f"model:{opp_200}", "model_opp")]),
        ],
        deck_pool=[{"main": [1]}],
        deck_paths=["deck.ydk"],
        manifest=manifest,
        sink=sink,
        num_episodes=10,
        seed=0,
        workers=1,
        agent_player="random",
        force=False,
        evaluate_fn=lambda **k: [EvalResult("ignored", 10, 6)],
        load_fn=lambda p, **k: {"global_step": checkpoint_update(p) * 2048},
    )
    assert summary["ok"] == 2
    tags = {(key, step) for key, _, step in sink.scalars}
    # Same tag, two points at the two checkpoints' global_steps.
    assert ("win_rate/model_opp/overall", 100 * 2048) in tags
    assert ("win_rate/model_opp/overall", 200 * 2048) in tags
    # Exactly one win-rate tag across both checkpoints (a curve, not one tag per N).
    win_rate_keys = {
        key
        for key, _, _ in sink.scalars
        if key.startswith("win_rate/") and key.endswith("/overall")
    }
    assert win_rate_keys == {"win_rate/model_opp/overall"}


def test_steps_per_update_reads_config_product():
    from cli.eval_sweep import _steps_per_update

    # config as a dict
    def dict_load(p, **k):
        return {"config": {"num_envs": 8, "rollout_steps": 256}}

    assert _steps_per_update([Path("run/checkpoint_1.pt")], load_fn=dict_load) == 8 * 256

    # config as an object (getattr path)
    class _C:
        num_envs, rollout_steps = 4, 128

    def obj_load(p, **k):
        return {"config": _C()}

    assert _steps_per_update([Path("run/checkpoint_1.pt")], load_fn=obj_load) == 4 * 128

    # No config -> None (caller then skips the mismatch warning).
    assert _steps_per_update([Path("x/checkpoint_1.pt")], load_fn=lambda p, **k: {}) is None


def test_parse_args_defaults():
    a = parse_args(["--run-dir", "r", "--opponents", "random"])
    assert a.run_dir == "r" and a.opponents == ["random"]
    assert a.stride == 1 and a.episodes == 1000 and a.workers == 1
    assert a.deck_paths is None and a.force is False


def test_parse_args_multi_opponents_and_stride():
    a = parse_args(
        ["--run-dir", "r", "--opponents", "random", "greedy", "--stride", "5", "--force"]
    )
    assert a.opponents == ["random", "greedy"] and a.stride == 5 and a.force is True


def test_parse_args_opponent_dir():
    a = parse_args(["--run-dir", "r", "--opponent-dir", "o"])
    assert a.opponent_dir == "o" and a.opponents is None


def test_main_no_checkpoints_returns_nonzero(tmp_path, capsys):
    from cli.eval_sweep import main

    rc = main(["--run-dir", str(tmp_path), "--opponents", "random"])
    assert rc == 2
    assert "no checkpoints" in capsys.readouterr().err.lower()
