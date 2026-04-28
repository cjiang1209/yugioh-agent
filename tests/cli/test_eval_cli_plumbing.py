"""In-process tests for cli.eval — argparse → eval.evaluate plumbing.

These tests run cli.eval.main() in-process with sys.argv mocked, patching
the heavy dependencies (parse_deck_pool, make_eval_agent, evaluate) to
verify only the CLI's plumbing — what kwargs get forwarded, how --device is
resolved, and how --json output is shaped.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cli import eval as cli_eval
from yugioh_rl.eval import EvalResult


@pytest.fixture
def stub_eval_pipeline():
    """Patch parse_deck_pool, make_eval_agent, evaluate. Return a captured-call
    dict for assertions and let the test customize evaluate's return value."""
    captured: dict = {"results": [
        EvalResult("greedy", episodes=2, wins=1, win_rate=0.5, per_deck_wins={0: [1.0, 0.0]}),
    ]}

    def fake_parse_deck_pool(paths):
        captured["deck_paths"] = list(paths)
        return [{"main": list(range(40)), "extra": []} for _ in paths]

    def fake_make_eval_agent(spec, *, seed=0, device="cpu", network=None):
        captured["agent_spec"] = spec
        captured["agent_seed"] = seed
        captured["agent_device"] = device
        return object()

    def fake_evaluate(agent, **kwargs):
        captured["evaluate_agent"] = agent
        captured.update(kwargs)
        return captured["results"]

    with patch("yugioh_rl.env_wrapper.parse_deck_pool", fake_parse_deck_pool), \
         patch("yugioh_rl.eval.make_eval_agent", fake_make_eval_agent), \
         patch("yugioh_rl.eval.evaluate", fake_evaluate):
        yield captured


# ---------------------------------------------------------------------------
# Kwarg forwarding
# ---------------------------------------------------------------------------

def test_forwards_args_to_evaluate(stub_eval_pipeline, deck_path_str):
    cli_eval.main([
        "--agent", "greedy",
        "--opponents", "random", "greedy",
        "--deck-paths", deck_path_str,
        "--episodes", "5",
        "--seed", "7",
        "--agent-player", "first",
    ])

    cap = stub_eval_pipeline
    assert cap["agent_spec"] == "greedy"
    assert cap["agent_seed"] == 7
    assert cap["opponent_specs"] == ["random", "greedy"]
    assert cap["num_episodes"] == 5
    assert cap["seed"] == 7
    assert cap["agent_player"] == "first"
    # Deck pool was parsed from --deck-paths
    assert cap["deck_paths"] == [deck_path_str]


# ---------------------------------------------------------------------------
# --device threads to both sides
# ---------------------------------------------------------------------------

def test_device_threads_to_both_agent_and_opponent(stub_eval_pipeline, deck_path_str):
    cli_eval.main([
        "--agent", "greedy",
        "--opponents", "greedy",
        "--deck-paths", deck_path_str,
        "--episodes", "1",
        "--device", "cpu",
    ])
    cap = stub_eval_pipeline
    assert cap["agent_device"] == "cpu"
    assert cap["opponent_device"] == "cpu"


def test_device_auto_resolved_before_forwarding(stub_eval_pipeline, deck_path_str):
    """--device auto must be resolved to a concrete cpu/cuda string."""
    with patch("cli.eval.resolve_device") as resolve_mock:
        resolve_mock.return_value = "cuda"
        cli_eval.main([
            "--agent", "greedy",
            "--opponents", "greedy",
            "--deck-paths", deck_path_str,
            "--episodes", "1",
            "--device", "auto",
        ])
    resolve_mock.assert_called_once_with("auto")
    cap = stub_eval_pipeline
    assert cap["agent_device"] == "cuda"
    assert cap["opponent_device"] == "cuda"


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------

def test_json_output_writes_expected_shape(stub_eval_pipeline, deck_path_str, tmp_path):
    out = tmp_path / "results.json"
    stub_eval_pipeline["results"] = [
        EvalResult("greedy", episodes=4, wins=3, win_rate=0.75,
                   per_deck_wins={0: [1.0, 1.0, 1.0, 0.0]}),
        EvalResult("random", episodes=4, wins=2, win_rate=0.5,
                   per_deck_wins={0: [1.0, 0.0, 1.0, 0.0]}),
    ]
    cli_eval.main([
        "--agent", "greedy",
        "--opponents", "greedy", "random",
        "--deck-paths", deck_path_str,
        "--episodes", "4",
        "--json", str(out),
    ])

    payload = json.loads(out.read_text())
    assert len(payload["opponents"]) == 2
    greedy = payload["opponents"][0]
    assert greedy["label"] == "greedy"
    assert greedy["episodes"] == 4
    assert greedy["wins"] == 3
    assert greedy["win_rate"] == 0.75
    assert "starter" in greedy["per_deck"]
    assert greedy["per_deck"]["starter"] == {"wins": 3, "episodes": 4, "win_rate": 0.75}


def test_no_json_flag_skips_file_write(stub_eval_pipeline, deck_path_str, capsys):
    cli_eval.main([
        "--agent", "greedy",
        "--opponents", "greedy",
        "--deck-paths", deck_path_str,
        "--episodes", "1",
    ])
    # No file written; stdout has the table but no JSON path message.
    out = capsys.readouterr().out
    assert "Wrote JSON results" not in out
    assert "vs greedy" in out


# ---------------------------------------------------------------------------
# Console table output
# ---------------------------------------------------------------------------

def test_console_table_includes_per_deck(stub_eval_pipeline, deck_path_str, capsys):
    cli_eval.main([
        "--agent", "greedy",
        "--opponents", "greedy",
        "--deck-paths", deck_path_str,
        "--episodes", "2",
    ])
    out = capsys.readouterr().out
    assert "vs greedy: 1/2 (50.0%)" in out
    assert "deck starter:" in out
