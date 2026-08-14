"""Tests for the numpy-backed encoding fields on YuGiOhObservation."""

import numpy as np
import pytest
from pydantic import ValidationError

from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    CHAIN_ENTRY_FEATURES,
    EVENT_ENTRY_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    MAX_EVENT_HISTORY,
    MAX_PENDING_CHAIN,
)
from yugioh_env.models import YuGiOhObservation

FIELDS = {
    "cards": ((MAX_CARDS, CARD_FEATURES), np.uint8),
    "global_state": ((GLOBAL_FEATURES,), np.uint8),
    "actions": ((MAX_ACTIONS, ACTION_FEATURES), np.uint8),
    "action_mask": ((MAX_ACTIONS,), np.int8),
    "pending_chain": ((MAX_PENDING_CHAIN, CHAIN_ENTRY_FEATURES), np.uint8),
    "event_history": ((MAX_EVENT_HISTORY, EVENT_ENTRY_FEATURES), np.uint8),
}


@pytest.mark.parametrize("name,spec", FIELDS.items())
def test_omitted_fields_default_to_shaped_numpy(name, spec) -> None:
    shape, dtype = spec
    arr = getattr(YuGiOhObservation(), name)  # ALL fields omitted
    assert isinstance(arr, np.ndarray)
    assert arr.shape == shape and arr.dtype == dtype


@pytest.mark.parametrize("name,spec", FIELDS.items())
@pytest.mark.parametrize("empty", [[], np.array([], dtype=np.uint8), None])
def test_empty_inputs_normalize_to_shaped_zeros(name, spec, empty) -> None:
    """Covers list, EMPTY NDARRAY, and None. The ndarray case is the one a
    naive validator misses: np.array([]) is already an ndarray, so an
    early `isinstance(v, np.ndarray): return v` would leak shape (0,)."""
    shape, dtype = spec
    arr = getattr(YuGiOhObservation(**{name: empty}), name)
    assert arr.shape == shape and arr.dtype == dtype
    assert not arr.any()


@pytest.mark.parametrize("name,spec", FIELDS.items())
def test_flattened_input_is_reshaped(name, spec) -> None:
    shape, dtype = spec
    flat = np.ones(int(np.prod(shape)), dtype=dtype)
    assert getattr(YuGiOhObservation(**{name: flat}), name).shape == shape


@pytest.mark.parametrize("name,spec", FIELDS.items())
def test_wrong_sized_input_raises(name, spec) -> None:
    shape, dtype = spec
    with pytest.raises(ValidationError):
        YuGiOhObservation(**{name: np.ones(int(np.prod(shape)) + 1, dtype=dtype)})


def test_transposed_same_size_2d_input_raises() -> None:
    """A right-SIZE, wrong-SHAPE 2-D input (e.g. cards transposed) has the
    same element count as the field shape, so a naive `reshape` would accept
    it silently and scramble the data instead of raising. This is the single
    contract-enforcement point for the whole observation -- it must be a
    loud failure, not a transpose."""
    shape, dtype = FIELDS["cards"]
    transposed = np.ones(shape[::-1], dtype=dtype)
    with pytest.raises(ValidationError):
        YuGiOhObservation(cards=transposed)


@pytest.mark.parametrize("name,spec", FIELDS.items())
def test_json_schema_nesting_depth_matches_shape(name, spec) -> None:
    shape, _ = spec
    node = YuGiOhObservation.model_json_schema()["properties"][name]
    depth = 0
    while node.get("type") == "array":
        node = node["items"]
        depth += 1
    assert depth == len(shape)


def test_json_wire_round_trip() -> None:
    obs = YuGiOhObservation()
    payload = obs.model_dump()
    assert isinstance(payload["actions"], list)  # python mode -> lists
    back = YuGiOhObservation(**payload)
    assert np.array_equal(back.as_arrays()["actions"], obs.as_arrays()["actions"])
