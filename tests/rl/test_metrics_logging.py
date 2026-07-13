from pathlib import Path

import pytest

from yugioh_rl.metrics_logging import (
    CheckpointEvent,
    CheckpointRef,
    MultiSink,
    ScalarMetrics,
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
        episode_length_mean=12.0,
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
    assert s["episode/length"] == 12.0
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
        "win_rate_vs_greedy": 0.8,
        "win_rate_vs_greedy_deck_blue_eyes": 0.7,
        "win_rate_vs_greedy_deck_exodia": 0.9,
    }
