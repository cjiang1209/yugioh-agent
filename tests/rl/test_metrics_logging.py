import types
from pathlib import Path

import pytest

from yugioh_rl.metrics_logging import (
    CheckpointEvent,
    CheckpointRef,
    MLflowSink,
    MultiSink,
    ScalarMetrics,
    TensorBoardSink,
    compute_update_metrics,
    flatten_eval,
)


class _RecordingSink:
    def __init__(self):
        self.events = []
        self.closed = False

    def handle(self, event):
        self.events.append(event)

    def close(self):
        self.closed = True


class _RaisingSink:
    def handle(self, event):
        raise RuntimeError("sink boom")

    def close(self):
        pass


def test_scalar_metrics_holds_scalars_and_step():
    m = ScalarMetrics(scalars={"loss/policy": 0.5}, global_step=100)
    assert m.scalars == {"loss/policy": 0.5}
    assert m.global_step == 100


def test_checkpoint_event_defaults_to_empty_scalars():
    ref = CheckpointRef(path=Path("/tmp/checkpoint_100.pt"), update=100, global_step=2048, tags={})
    ev = CheckpointEvent(ref=ref)
    assert ev.scalars == {}
    assert ev.ref.update == 100


def test_multisink_fans_out_to_all_sinks():
    a, b = _RecordingSink(), _RecordingSink()
    sink = MultiSink([a, b])
    m = ScalarMetrics(scalars={"x": 1.0}, global_step=1)
    sink.handle(m)
    assert a.events == [m]
    assert b.events == [m]


def test_multisink_close_closes_all():
    a, b = _RecordingSink(), _RecordingSink()
    MultiSink([a, b]).close()
    assert a.closed and b.closed


def test_multisink_is_fail_loud():
    sink = MultiSink([_RaisingSink()])
    with pytest.raises(RuntimeError, match="sink boom"):
        sink.handle(ScalarMetrics(scalars={}, global_step=0))


def test_compute_update_metrics_core_scalars():
    m = compute_update_metrics(
        global_step=2048,
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        fps=1500.0,
    )
    assert m.global_step == 2048
    assert m.scalars == {
        "loss/policy": 0.1,
        "loss/value": 0.2,
        "loss/entropy": 0.3,
        "perf/fps": 1500.0,
    }


def test_compute_update_metrics_optional_blocks():
    m = compute_update_metrics(
        global_step=10,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        fps=1.0,
        episode_reward_mean=0.5,
        episode_win_rate=0.6,
        episode_steps_mean=12.0,
        deck_win_rates={"blue_eyes": 0.7},
        elo={
            "agent": 1500.0,
            "pool_mean": 1400.0,
            "pool_min": 1300.0,
            "pool_max": 1600.0,
            "occupied": 3,
        },
        async_stats={"version_lag_mean": 1.5, "rollouts_discarded": 2, "queue_depth": 4},
    )
    s = m.scalars
    assert s["episode/reward"] == 0.5
    assert s["episode/win_rate"] == 0.6
    assert s["episode/steps"] == 12.0
    assert s["episode/win_rate_deck_blue_eyes"] == 0.7
    assert s["selfplay/elo_agent"] == 1500.0
    assert s["selfplay/elo_pool_mean"] == 1400.0
    assert s["selfplay/elo_pool_min"] == 1300.0
    assert s["selfplay/elo_pool_max"] == 1600.0
    assert s["selfplay/occupied"] == 3
    assert s["async/version_lag_mean"] == 1.5
    assert s["async/rollouts_discarded"] == 2
    assert s["async/queue_depth"] == 4


def test_compute_update_metrics_omits_queue_depth_when_absent():
    m = compute_update_metrics(
        global_step=10,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        fps=1.0,
        async_stats={"version_lag_mean": 0.0, "rollouts_discarded": 0},
    )
    assert "async/queue_depth" not in m.scalars


def test_flatten_eval_keys_match_tb_convention():
    row = {
        "win_rate": 0.8,
        "per_deck": {"blue_eyes": {"win_rate": 0.7}, "exodia": {"win_rate": 0.9}},
    }
    out = flatten_eval(row, "greedy")
    assert out == {
        "win_rate/greedy/overall": 0.8,
        "win_rate_by_deck/greedy/blue_eyes": 0.7,
        "win_rate_by_deck/greedy/exodia": 0.9,
    }


def test_flatten_eval_new_scheme():
    row = {
        "win_rate": 0.75,
        "wins": 3,
        "episodes": 4,
        "per_deck": {"blue_eyes": {"wins": 2, "episodes": 3, "win_rate": 0.667}},
        "steps": {"mean": 25.0, "std": 1.0, "median": 25.0, "max": 40},
        "turns": {"mean": 7.0, "std": 2.0, "median": 7.0, "max": 10},
        "play_first_rate": 0.5,
        "wins_first": 2,
        "episodes_first": 2,
        "wins_second": 1,
        "episodes_second": 2,
    }
    out = flatten_eval(row, "opp")
    assert out["win_rate/opp/overall"] == 0.75
    assert out["win_rate_by_deck/opp/blue_eyes"] == 0.667
    assert out["steps/opp/mean"] == 25.0 and out["steps/opp/max"] == 40
    assert out["turns/opp/median"] == 7.0 and out["play_first_rate/opp"] == 0.5
    assert out["win_rate/opp/play_first"] == 1.0 and out["win_rate/opp/play_second"] == 0.5


def test_in_training_eval_scalars_prefixed_new_scheme():
    from yugioh_rl.eval import EvalResult
    from yugioh_rl.ppo import _eval_scalars

    r = EvalResult(
        opponent_label="random",
        episodes=2,
        wins=1,
        per_deck_wins={0: [1.0, 0.0]},
        steps_mean=3.0,
        steps_std=0.0,
        steps_median=3.0,
        steps_max=3,
        turns_mean=2.0,
        turns_std=0.0,
        turns_median=2.0,
        turns_max=2,
        wins_first=1,
        episodes_first=2,
        wins_second=0,
        episodes_second=0,
    )
    s = _eval_scalars([r], ["decks/blue_eyes.ydk"])
    assert s["eval/win_rate/random/overall"] == 0.5
    assert s["eval/win_rate_by_deck/random/blue_eyes"] == 0.5
    assert s["eval/steps/random/mean"] == 3.0 and s["eval/play_first_rate/random"] == 1.0
    assert s["eval/win_rate/random/play_first"] == 0.5
    assert "eval/win_rate/random/play_second" not in s


def test_flatten_eval_omits_empty_split():
    row = {
        "win_rate": 0.6,
        "wins": 6,
        "episodes": 10,
        "per_deck": {},
        "steps": {"mean": 1, "std": 0, "median": 1, "max": 1},
        "turns": {"mean": 1, "std": 0, "median": 1, "max": 1},
        "play_first_rate": 1.0,
        "wins_first": 6,
        "episodes_first": 10,
        "wins_second": 0,
        "episodes_second": 0,
    }
    out = flatten_eval(row, "opp")
    assert "win_rate/opp/play_first" in out and "win_rate/opp/play_second" not in out


class _FakeWriter:
    def __init__(self):
        self.calls = []
        self.closed = False

    def add_scalar(self, key, value, step):
        self.calls.append((key, value, step))

    def close(self):
        self.closed = True


def test_tb_sink_scalar_metrics_no_prefix():
    w = _FakeWriter()
    TensorBoardSink(w).handle(ScalarMetrics(scalars={"loss/policy": 0.5}, global_step=100))
    assert w.calls == [("loss/policy", 0.5, 100)]


def test_tb_sink_checkpoint_event_scalars_unprefixed_at_global_step():
    w = _FakeWriter()
    ref = CheckpointRef(path=Path("/tmp/checkpoint_5.pt"), update=5, global_step=999, tags={})
    ev = CheckpointEvent(ref=ref, scalars={"win_rate/greedy/overall": 0.8})
    TensorBoardSink(w).handle(ev)
    assert w.calls == [("win_rate/greedy/overall", 0.8, 999)]


def test_tb_sink_eval_scalars_unprefixed():
    w = _FakeWriter()
    ref = CheckpointRef(path=Path("checkpoint_100.pt"), update=100, global_step=2048, tags={})
    TensorBoardSink(w).handle(CheckpointEvent(ref=ref, scalars={"win_rate/opp/overall": 0.6}))
    keys = [k for k, _, _ in w.calls]
    assert "win_rate/opp/overall" in keys and "eval/win_rate/opp/overall" not in keys


def test_tb_sink_registration_event_is_noop():
    w = _FakeWriter()
    ref = CheckpointRef(path=Path("/tmp/checkpoint_5.pt"), update=5, global_step=1, tags={})
    TensorBoardSink(w).handle(CheckpointEvent(ref=ref))  # empty scalars
    assert w.calls == []


def test_tb_sink_close_closes_writer():
    w = _FakeWriter()
    TensorBoardSink(w).close()
    assert w.closed


class _FakeLoggedModel:
    def __init__(self, model_id):
        self.model_id = model_id


class _FakeMlflow:
    def __init__(self, existing=None):
        self._existing = existing or []
        self.metrics = []  # (key, value, step, model_id)
        self.created = []  # (name, tags)
        self.artifacts = []  # (local_path, artifact_path)
        self.ended = False
        self.system_metrics_enabled = False

    def enable_system_metrics_logging(self):
        self.system_metrics_enabled = True

    def log_metrics(self, metrics, step=None, model_id=None):
        for key, value in metrics.items():
            self.metrics.append((key, value, step, model_id))

    def search_logged_models(self, filter_string=None, output_format=None):
        assert output_format == "list"
        return list(self._existing)

    def create_external_model(self, name=None, tags=None, params=None):
        self.created.append((name, tags, params))
        lm = _FakeLoggedModel(model_id=f"m-{name}")
        self._existing = [lm]
        return lm

    def log_artifact(self, local_path, artifact_path=None):
        self.artifacts.append((local_path, artifact_path))

    def end_run(self):
        self.ended = True


def _write_ckpt(tmp_path, name="checkpoint_100.pt", data=b"weights"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_mlflow_sink_scalar_metrics_go_to_run():
    fake = _FakeMlflow()
    MLflowSink(fake).handle(ScalarMetrics(scalars={"loss/policy": 0.5}, global_step=100))
    assert fake.metrics == [("loss/policy", 0.5, 100, None)]


def test_mlflow_sink_registration_creates_model_and_uploads(tmp_path):
    fake = _FakeMlflow(existing=[])
    p = _write_ckpt(tmp_path)
    ref = CheckpointRef(
        path=p,
        update=100,
        global_step=2048,
        params={"seed": "42"},
        tags={"note": "manual"},  # optional arbitrary metadata; passed through verbatim
    )
    MLflowSink(fake).handle(CheckpointEvent(ref=ref))  # empty scalars
    assert len(fake.created) == 1
    name, tags, params = fake.created[0]
    assert name == "checkpoint_100"
    # tags: caller metadata passed through untouched
    assert tags == {"note": "manual"}
    # params: caller facts + sink-added defining facts (incl. the searchable hash join key)
    assert params["seed"] == "42"
    assert params["update"] == "100"
    assert params["global_step"] == "2048"
    assert len(params["checkpoint_hash"]) == 64
    # defining facts are params, NOT tags
    assert "checkpoint_hash" not in tags
    assert "update" not in tags
    assert "global_step" not in tags
    assert fake.artifacts == [(str(p), "checkpoints/checkpoint_100")]
    assert fake.metrics == []  # empty scalars -> no metric attached


def test_mlflow_sink_eval_attaches_metrics_to_existing_model(tmp_path):
    existing = _FakeLoggedModel(model_id="m-existing")
    fake = _FakeMlflow(existing=[existing])
    p = _write_ckpt(tmp_path)
    ref = CheckpointRef(path=p, update=100, global_step=2048, tags={})
    ev = CheckpointEvent(ref=ref, scalars={"win_rate/greedy/overall": 0.8})
    MLflowSink(fake).handle(ev)
    assert fake.created == []  # found existing -> no create
    assert fake.metrics == [("win_rate/greedy/overall", 0.8, 2048, "m-existing")]


def test_mlflow_sink_close_ends_run():
    fake = _FakeMlflow()
    MLflowSink(fake).close()
    assert fake.ended


def test_build_training_sinks_tensorboard_only(tmp_path):
    from yugioh_rl.metrics_logging import build_training_sinks

    sink = build_training_sinks(
        log_to=["tensorboard"],
        save_dir=str(tmp_path),
        purge_step=None,
        params={},
    )
    assert isinstance(sink, MultiSink)
    assert len(sink._sinks) == 1
    sink.close()


def test_build_training_sinks_mlflow_missing_uri_hard_fails(tmp_path, monkeypatch):
    from yugioh_rl import metrics_logging

    fake = _FakeMlflow()  # _require_tracking_uri raises before any fake method is reached
    monkeypatch.setattr(metrics_logging, "_import_mlflow", lambda: fake)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        metrics_logging.build_training_sinks(
            log_to=["mlflow"],
            save_dir=str(tmp_path),
            purge_step=None,
            params={},
        )


def test_build_training_sinks_mlflow_writes_run_id(tmp_path, monkeypatch):
    from yugioh_rl import metrics_logging

    started = {}
    fake = _FakeMlflow()
    fake.set_tracking_uri = lambda uri: started.setdefault("uri", uri)
    fake.set_experiment = lambda name: started.setdefault("exp", name)

    def _start_run(**kwargs):
        started["start_kwargs"] = kwargs
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-xyz"))

    fake.start_run = _start_run
    fake.log_params = lambda p: started.setdefault("params", p)
    monkeypatch.setattr(metrics_logging, "_import_mlflow", lambda: fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")

    (tmp_path / "config.json").write_text('{"seed": 42}')  # run's config snapshot

    sink = metrics_logging.build_training_sinks(
        log_to=["mlflow"],
        save_dir=str(tmp_path),
        purge_step=None,
        params={"seed": "42"},
    )
    assert (tmp_path / "mlflow_run_id.txt").read_text().strip() == "run-xyz"
    assert started["exp"] == "yugioh"
    assert started["params"] == {"seed": "42"}
    assert started["start_kwargs"].get("run_id") is None  # fresh run uses run_name
    assert started["start_kwargs"]["run_name"] == f"train_{tmp_path.name}"  # symmetric w/ eval_
    assert fake.system_metrics_enabled  # hardware telemetry turned on
    assert fake.artifacts == [(str(tmp_path / "config.json"), None)]  # config snapshot uploaded
    sink.close()


def test_build_training_sinks_reattaches_existing_run_id(tmp_path, monkeypatch):
    from yugioh_rl import metrics_logging

    (tmp_path / "mlflow_run_id.txt").write_text("prev-run")
    started = {}
    fake = _FakeMlflow()
    fake.set_tracking_uri = lambda uri: None
    fake.set_experiment = lambda name: None

    def _start_run(**kwargs):
        started["start_kwargs"] = kwargs
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="prev-run"))

    fake.start_run = _start_run
    fake.log_params = lambda p: started.setdefault("params", p)
    monkeypatch.setattr(metrics_logging, "_import_mlflow", lambda: fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")

    metrics_logging.build_training_sinks(
        log_to=["mlflow"],
        save_dir=str(tmp_path),
        purge_step=None,
        params={"seed": "42"},
    )
    assert started["start_kwargs"]["run_id"] == "prev-run"
    # Params must NOT be re-logged on reattach: re-logging a changed value
    # (e.g. resume_checkpoint) makes MLflow raise and kills --resume.
    assert "params" not in started


def test_build_eval_sinks_opens_named_run(tmp_path, monkeypatch):
    from yugioh_rl import metrics_logging

    started = {}
    fake = _FakeMlflow()
    fake.set_tracking_uri = lambda uri: None
    fake.set_experiment = lambda name: started.setdefault("exp", name)
    fake.start_run = lambda **k: (
        started.setdefault("start_kwargs", k)
        or types.SimpleNamespace(info=types.SimpleNamespace(run_id="e1"))
    )
    fake.log_params = lambda p: started.setdefault("params", p)
    monkeypatch.setattr(metrics_logging, "_import_mlflow", lambda: fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")

    run_dir = tmp_path / "20260713_120000_seed42"
    run_dir.mkdir()
    sink = metrics_logging.build_eval_sinks(
        log_to=["mlflow"],
        run_dir=str(run_dir),
        params={"opponents": "greedy,random", "episodes": "1000"},
    )
    assert started["exp"] == "yugioh"
    assert started["start_kwargs"]["run_name"] == "eval_20260713_120000_seed42"
    assert started["params"] == {"opponents": "greedy,random", "episodes": "1000"}
    assert fake.system_metrics_enabled  # hardware telemetry on for sweeps too
    sink.close()


def test_build_eval_sinks_custom_subdir_and_run_name(tmp_path, monkeypatch):
    """Step-matched evaluation lands in its own TB board + MLflow run."""
    from yugioh_rl import metrics_logging

    started = {}
    fake = _FakeMlflow()
    fake.set_tracking_uri = lambda uri: None
    fake.set_experiment = lambda name: None
    fake.start_run = lambda **k: (
        started.setdefault("start_kwargs", k)
        or types.SimpleNamespace(info=types.SimpleNamespace(run_id="e2"))
    )
    fake.log_params = lambda p: None
    monkeypatch.setattr(metrics_logging, "_import_mlflow", lambda: fake)
    # Capture the TB log dir without a real SummaryWriter (keep this file torch-free).
    monkeypatch.setattr(
        metrics_logging,
        "_new_tb_writer",
        lambda log_dir, purge_step: (started.update({"tb_dir": log_dir}), _FakeWriter())[1],
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")

    sink = metrics_logging.build_eval_sinks(
        log_to=["tensorboard", "mlflow"],
        run_dir=str(tmp_path / "runA"),
        params={},
        subdir="eval_vs_runB",
        run_name="eval_runA_vs_runB",
    )
    assert started["start_kwargs"]["run_name"] == "eval_runA_vs_runB"
    assert started["tb_dir"].endswith("runA/logs/eval_vs_runB")  # own TB board dir
    sink.close()
