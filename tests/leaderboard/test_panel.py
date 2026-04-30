"""Tests for the leaderboard panel config."""

from __future__ import annotations

import json

import pytest

from yugioh_leaderboard.panel import (
    PanelConfig,
    PanelEntry,
    PanelMatchOptions,
    load_panel_config,
)


def _sample_dict() -> dict:
    return {
        "schema_version": 1,
        "panel_version": 2,
        "panel": [
            {"label": "random", "spec": "random"},
            {"label": "greedy", "spec": "greedy"},
        ],
        "match": {"episodes": 100, "agent_player": "random", "device": "cpu"},
        "history": [
            {"panel_version": 1, "panel": [{"label": "random", "spec": "random"}],
             "retired_at": "2026-03-01T00:00:00Z"},
        ],
    }


def test_load_valid_config(tmp_path):
    p = tmp_path / "leaderboard.config.json"
    p.write_text(json.dumps(_sample_dict()))
    cfg = load_panel_config(p)
    assert cfg.panel_version == 2
    assert len(cfg.panel) == 2
    assert cfg.panel[0].label == "random"
    assert cfg.panel[0].spec == "random"
    assert cfg.match.episodes == 100
    assert cfg.match.agent_player == "random"
    assert len(cfg.history) == 1


def test_missing_panel_key_rejected(tmp_path):
    d = _sample_dict()
    del d["panel"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="panel"):
        load_panel_config(p)


def test_missing_panel_version_rejected(tmp_path):
    d = _sample_dict()
    del d["panel_version"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="panel_version"):
        load_panel_config(p)


def test_unknown_opponent_kind_rejected(tmp_path):
    d = _sample_dict()
    d["panel"][0]["spec"] = "weirdthing"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="unknown opponent"):
        load_panel_config(p)


def test_model_spec_without_path_rejected(tmp_path):
    d = _sample_dict()
    d["panel"][0]["spec"] = "model:"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="checkpoint path"):
        load_panel_config(p)


def test_match_block_missing_episodes_rejected(tmp_path):
    d = _sample_dict()
    del d["match"]["episodes"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="episodes"):
        load_panel_config(p)


def test_invalid_agent_player_rejected(tmp_path):
    d = _sample_dict()
    d["match"]["agent_player"] = "upside_down"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="agent_player"):
        load_panel_config(p)


def test_invalid_device_rejected(tmp_path):
    d = _sample_dict()
    d["match"]["device"] = "tpu"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="device"):
        load_panel_config(p)


def test_duplicate_panel_labels_rejected(tmp_path):
    d = _sample_dict()
    d["panel"] = [
        {"label": "r", "spec": "random"},
        {"label": "r", "spec": "greedy"},
    ]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="duplicate label"):
        load_panel_config(p)
