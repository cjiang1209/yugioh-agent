"""End-to-end eval-sweep smoke test. Skips without torch/libocgcore/cdb."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _write_checkpoint(path: Path, update: int, global_step: int, deck_path: str):
    """Write a minimal real checkpoint the sweep can eval (symbolic net)."""
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import YuGiOhNet

    config = TrainingConfig(deck_paths=[deck_path])
    net = YuGiOhNet(config)
    torch.save(
        {
            "update": update,
            "global_step": global_step,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
        },
        path,
    )


def test_eval_sweep_end_to_end(tmp_path, lib, db_path, script_dirs, deck_path):
    from cli.eval_sweep import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir / "checkpoint_1.pt", 1, 100, str(deck_path))
    _write_checkpoint(run_dir / "checkpoint_2.pt", 2, 200, str(deck_path))

    json_out = tmp_path / "summary.json"
    rc = main(
        [
            "--run-dir",
            str(run_dir),
            "--opponents",
            "random",
            "--episodes",
            "2",
            "--workers",
            "1",
            "--seed",
            "0",
            "--json",
            str(json_out),
        ]
    )
    assert rc == 0

    # manifest records both checkpoints vs random
    manifest = json.loads((run_dir / "logs" / "eval" / "manifest.json").read_text())
    pairs = {(r["update"], r["label"]) for r in manifest["results"]}
    assert (1, "random") in pairs and (2, "random") in pairs

    # a TensorBoard event file was written under logs/eval/
    events = list((run_dir / "logs" / "eval").glob("events.out.tfevents.*"))
    assert events, "expected a TensorBoard event file in logs/eval/"

    # JSON summary written with ok==2
    summary = json.loads(json_out.read_text())
    assert summary["ok"] == 2 and summary["failed"] == 0


def test_eval_sweep_resume_skips_done(tmp_path, lib, db_path, script_dirs, deck_path):
    from cli.eval_sweep import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(run_dir / "checkpoint_1.pt", 1, 100, str(deck_path))

    main(
        [
            "--run-dir",
            str(run_dir),
            "--opponents",
            "random",
            "--episodes",
            "2",
            "--workers",
            "1",
            "--seed",
            "0",
        ]
    )
    # second run: the one pair is already recorded → skipped, not re-evaluated
    json_out = tmp_path / "summary2.json"
    main(
        [
            "--run-dir",
            str(run_dir),
            "--opponents",
            "random",
            "--episodes",
            "2",
            "--workers",
            "1",
            "--seed",
            "0",
            "--json",
            str(json_out),
        ]
    )
    summary = json.loads(json_out.read_text())
    assert summary["skipped"] == 1 and summary["ok"] == 0
