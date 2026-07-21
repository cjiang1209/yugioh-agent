"""Tests for build_forward_inputs — the shared obs-dict -> forward-kwargs helper."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from yugioh_rl.policy_inputs import build_forward_inputs

_EXPECTED_KEYS = {
    "obs_cards",
    "obs_global",
    "obs_actions",
    "action_mask",
    "obs_chain",
    "obs_event",
}


def _obs() -> dict:
    return {
        "cards": np.zeros((200, 42), dtype=np.uint8),
        "global_state": np.zeros((21,), dtype=np.uint8),
        "actions": np.zeros((32, 28), dtype=np.uint8),
        "action_mask": np.ones((32,), dtype=np.int8),
        "pending_chain": np.zeros((8, 16), dtype=np.uint8),
        "event_history": np.zeros((32, 30), dtype=np.uint8),
    }


def test_returns_all_forward_kwargs() -> None:
    inputs = build_forward_inputs(_obs())
    assert set(inputs) == _EXPECTED_KEYS


def test_no_batch_dim_preserves_shapes() -> None:
    inputs = build_forward_inputs(_obs(), add_batch_dim=False)
    assert tuple(inputs["obs_cards"].shape) == (200, 42)
    assert tuple(inputs["action_mask"].shape) == (32,)
    assert tuple(inputs["obs_chain"].shape) == (8, 16)


def test_add_batch_dim_prepends_leading_one() -> None:
    inputs = build_forward_inputs(_obs(), add_batch_dim=True)
    assert tuple(inputs["obs_cards"].shape) == (1, 200, 42)
    assert tuple(inputs["action_mask"].shape) == (1, 32)
    assert tuple(inputs["obs_event"].shape) == (1, 32, 30)
    assert tuple(inputs["obs_chain"].shape) == (1, 8, 16)


def test_dtype_preserved() -> None:
    inputs = build_forward_inputs(_obs())
    assert inputs["obs_cards"].dtype == torch.uint8
    assert inputs["action_mask"].dtype == torch.int8


def test_guard_optional_yields_none_for_absent_keys() -> None:
    obs = _obs()
    del obs["pending_chain"]
    del obs["event_history"]
    inputs = build_forward_inputs(obs, guard_optional=True)
    assert inputs["obs_chain"] is None
    assert inputs["obs_event"] is None
    # Core keys are still built.
    assert inputs["obs_cards"] is not None
    assert inputs["action_mask"] is not None


def test_guard_optional_still_converts_present_keys() -> None:
    inputs = build_forward_inputs(_obs(), guard_optional=True, add_batch_dim=True)
    assert tuple(inputs["obs_chain"].shape) == (1, 8, 16)
    assert tuple(inputs["obs_event"].shape) == (1, 32, 30)


def test_direct_indexing_raises_on_absent_optional_key() -> None:
    obs = _obs()
    del obs["pending_chain"]
    # guard_optional=False (collection): the key is contractually always present,
    # so a missing key is a real error, not silently None.
    with pytest.raises(KeyError):
        build_forward_inputs(obs, guard_optional=False)


def test_strict_mode_raises_on_none_value() -> None:
    obs = _obs()
    obs["pending_chain"] = None
    # guard_optional=False: a present-but-None value must fail loudly (never
    # silently become None) — only the guarded serving path tolerates absence.
    with pytest.raises(TypeError):
        build_forward_inputs(obs, guard_optional=False)
