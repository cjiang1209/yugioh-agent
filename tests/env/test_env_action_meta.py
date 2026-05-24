"""Tests for ActionMeta and YuGiOhObservation.action_meta.

Trivial constructor / default-value tests are intentionally omitted — they
test Pydantic, not our code. We only assert non-trivial contracts:

1. The `kind` Literal actually rejects unknown strings (defends against
   Pydantic version drift silently downgrading validation).
2. action_meta serializes through model_dump → model_validate without
   losing None entries or downgrading nested ActionMeta dicts. This is the
   contract the web UI binds to over HTTP.
"""

import pytest
from pydantic import ValidationError

from yugioh_env.models import ActionMeta, YuGiOhObservation


def test_action_meta_rejects_bad_kind():
    with pytest.raises(ValidationError):
        ActionMeta(kind="bogus_kind", label="x")


def test_action_meta_round_trips_through_pydantic_serialization():
    """The HTTP transport contract: dump → restore preserves None entries
    and nested ActionMeta structure."""
    obs = YuGiOhObservation(
        action_meta=[
            ActionMeta(kind="number", label="Announce 3", raw_value=3),
            None,
            ActionMeta(
                kind="counter",
                label="Remove 2 from Card999",
                raw_value=1,
                extras={"counter_count": 2, "card_code": 999},
            ),
        ]
    )
    dumped = obs.model_dump()
    assert dumped["action_meta"][1] is None  # None survives serialization
    restored = YuGiOhObservation.model_validate(dumped)
    assert restored.action_meta[0].kind == "number"
    assert restored.action_meta[1] is None
    assert restored.action_meta[2].extras["card_code"] == 999  # extras dict preserved
