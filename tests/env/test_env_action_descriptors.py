"""Tests for YuGiOhObservation.action_descriptors.

Trivial constructor / default-value tests are intentionally omitted — they
test Pydantic, not our code. We only assert non-trivial contracts:

1. The `kind` discriminator actually rejects unknown strings (defends against
   Pydantic version drift silently downgrading validation) when embedded in
   a real YuGiOhObservation, not just the bare TypeAdapter.
2. action_descriptors serializes through model_dump -> model_validate
   without losing None entries or downgrading nested variant structure.
   This is the contract the web UI binds to over HTTP.
"""

import pytest
from pydantic import ValidationError

from yugioh_env.models import AnnounceNumber, CardRef, SelectCounter, YuGiOhObservation


def test_action_descriptor_rejects_bad_kind():
    with pytest.raises(ValidationError):
        YuGiOhObservation(action_descriptors=[{"kind": "bogus_kind"}])


def test_action_descriptors_round_trip_through_pydantic_serialization():
    """The HTTP transport contract: dump -> restore preserves None entries
    and nested variant structure."""
    obs = YuGiOhObservation(
        action_descriptors=[
            AnnounceNumber(engine_index=0, value=3),
            None,
            SelectCounter(
                engine_index=0,
                card=CardRef(code=999, controller=0, location=0x4, sequence=0),
                counter_type=1,
                counter_count=2,
            ),
        ]
    )
    dumped = obs.model_dump()
    assert dumped["action_descriptors"][1] is None  # None survives serialization
    restored = YuGiOhObservation.model_validate(dumped)
    assert restored.action_descriptors[0].kind == "announce_number"
    assert restored.action_descriptors[0].value == 3
    assert restored.action_descriptors[1] is None
    assert restored.action_descriptors[2].card.code == 999  # nested CardRef preserved
