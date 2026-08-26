# tests/mud/test_action_layout_parity.py
"""MUD's action rows must mean the same thing as the engine's.

The MUD goldens pin these rows against their own past, so a row can drift away
from `ACTION_LAYOUT` without failing them. These tests compare a MUD row
against the in-process row for the same action.

The text protocol reaches less than the engine does, so parity is not equality
on every prompt: `_ENGINE_ONLY` names the fields MUD leaves zero, and those
zeros are asserted. Filling one later fails here, so it lands as a deliberate
update.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.env.conftest import obs_from_msg
from tests.mud.conftest import build_cards_db
from yugioh_core.action_categories import IDLE_ACTIVATE, IDLE_SUMMON
from yugioh_core.card_database import CardDatabase
from yugioh_core.constants import LOCATION_HAND, LOCATION_MZONE, MSG_SELECT_IDLECMD
from yugioh_core.encoding import ACTION_LAYOUT
from yugioh_mud.cmd_handler import StructuredAction
from yugioh_mud.game_state import CardEntry, MUDGameState
from yugioh_mud.observation import MUDObservationBuilder
from yugioh_mud.text_parser import ParsedPrompt, PromptType
from yugioh_rl.obs_encoder import encode_observation

_CODE = 89631139

# Fields the engine fills from the message and the MUD text parser cannot
# reach. They stay zero on a MUD row.
_ENGINE_ONLY = (
    "desc",
    "position",
    "direct_attackable",
    "subsequence",
    "param",
    "counter_type",
    "counter_count",
)


@pytest.fixture(scope="module")
def mud_builder(tmp_path_factory: pytest.TempPathFactory) -> MUDObservationBuilder:
    db: Path = build_cards_db(
        tmp_path_factory.mktemp("parity") / "cards.cdb",
        [(_CODE, "Blue-Eyes White Dragon", 17, 3000, 2500, 8, 0x20, 0x10)],
    )
    return MUDObservationBuilder(CardDatabase(db))


def _mud_idle_row(mud_builder: MUDObservationBuilder, category: int) -> np.ndarray:
    """The MUD row for one idle-menu action on a monster in hand slot 0."""
    gs = MUDGameState()
    gs.my_lp, gs.opp_lp = 8000, 8000
    gs.turn, gs.phase, gs.is_my_turn = 1, "main1 phase", True
    gs.my_hand = [CardEntry(name="Blue-Eyes White Dragon", code=_CODE)]
    prompt = ParsedPrompt(
        prompt_type=PromptType.IDLE_CMD,
        options=["e"],
        structured_actions=[
            StructuredAction(
                category=category,
                cardspec="h1",
                card_code=_CODE,
                location=LOCATION_HAND,
                sequence=0,
                sub_action="s",
            )
        ],
    )
    return mud_builder.build(gs, prompt)["actions"][0]


def _engine_idle_row(key: str, entry: dict) -> np.ndarray:
    """The in-process row for a one-action idle prompt."""
    msg = {"msg_type": MSG_SELECT_IDLECMD, "player": 0, key: [entry]}
    return encode_observation(obs_from_msg(msg))["actions"][0]


def _fields(row: np.ndarray) -> dict[str, int]:
    return {name: ACTION_LAYOUT.read(row, name) for name, _ in ACTION_LAYOUT}


def test_summon_row_is_identical_to_the_engines(mud_builder) -> None:
    """A summon is the case both encoders can express fully, so the rows agree
    field for field -- the engine leaves the same fields zero that MUD cannot
    reach."""
    mud = _mud_idle_row(mud_builder, IDLE_SUMMON)
    engine = _engine_idle_row(
        "summonable",
        {"code": _CODE, "controller": 0, "location": LOCATION_HAND, "sequence": 0},
    )
    assert _fields(mud) == _fields(engine)


def test_activation_row_agrees_on_what_the_parser_reaches(mud_builder) -> None:
    """An activation is the case they diverge on: the engine carries the effect
    id and the activating card's position, neither of which appears in the MUD
    text."""
    mud = _mud_idle_row(mud_builder, IDLE_ACTIVATE)
    engine = _engine_idle_row(
        "activatable",
        {
            "code": _CODE,
            "controller": 0,
            "location": LOCATION_MZONE,
            "sequence": 2,
            "desc": 0x99,
            "client_mode": 0,
        },
    )
    shared = ("msg_type", "category", "code", "controller", "index", "num_selected")
    assert {f: ACTION_LAYOUT.read(mud, f) for f in shared} == {
        f: ACTION_LAYOUT.read(engine, f) for f in shared
    }
    # The engine fills these; MUD cannot, and their being zero is the gap.
    assert any(ACTION_LAYOUT.read(engine, f) for f in ("desc", "position"))
    assert all(ACTION_LAYOUT.read(mud, f) == 0 for f in _ENGINE_ONLY)
