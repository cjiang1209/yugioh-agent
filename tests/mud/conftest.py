"""Shared fixtures for the MUD test suite."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path


def build_cards_db(
    db_path: Path, cards: list[tuple[int, str, int, int, int, int, int, int]]
) -> Path:
    """Write a minimal cards.cdb holding just `cards` and return its path.

    Each tuple is (id, name, type, atk, def, level, race, attribute). The
    schema mirrors the real cards.cdb columns the card database reads;
    everything not listed keeps its default, so a card needs only the fields
    a test actually asserts on.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE datas ("
        "id INTEGER PRIMARY KEY, ot INTEGER DEFAULT 0, alias INTEGER DEFAULT 0, "
        "setcode INTEGER DEFAULT 0, type INTEGER DEFAULT 0, "
        "atk INTEGER DEFAULT 0, def INTEGER DEFAULT 0, "
        "level INTEGER DEFAULT 0, race INTEGER DEFAULT 0, "
        "attribute INTEGER DEFAULT 0)"
    )
    conn.execute("CREATE TABLE texts (id INTEGER PRIMARY KEY, name TEXT, desc TEXT)")
    for cid, name, ctype, atk, dfn, level, race, attr in cards:
        conn.execute(
            "INSERT INTO datas (id, type, atk, def, level, race, attribute) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, ctype, atk, dfn, level, race, attr),
        )
        conn.execute("INSERT INTO texts (id, name, desc) VALUES (?, ?, ?)", (cid, name, ""))
    conn.commit()
    conn.close()
    return db_path


def mud_observation_cases() -> dict[str, Callable[[], dict]]:
    """Named builders for MUD observations, shared by the golden capture
    script and the golden test so both pin exactly the same cases.

    Builds its own tiny cards.cdb rather than taking a pytest fixture, since
    the capture script that also calls this runs outside pytest.
    """
    from yugioh_core.action_categories import BATTLE_ATTACK, BATTLE_TO_EP, IDLE_SUMMON, IDLE_TO_EP
    from yugioh_core.card_database import CardDatabase
    from yugioh_core.constants import LOCATION_HAND, LOCATION_MZONE
    from yugioh_mud.cmd_handler import StructuredAction
    from yugioh_mud.game_state import CardEntry, MUDGameState
    from yugioh_mud.observation import MUDObservationBuilder
    from yugioh_mud.text_parser import ParsedPrompt, PromptType

    db_path = build_cards_db(
        Path(tempfile.mkdtemp()) / "cards.cdb",
        [
            (89631139, "Blue-Eyes White Dragon", 17, 3000, 2500, 8, 0x20, 0x10),
            (44095762, "Mirror Force", 4, 0, 0, 0, 0, 0),
        ],
    )

    card_db = CardDatabase(db_path)
    builder = MUDObservationBuilder(card_db)

    def _idle() -> dict:
        gs = MUDGameState()
        gs.my_lp, gs.opp_lp = 8000, 8000
        gs.turn = 1
        gs.phase = "main1 phase"
        gs.is_my_turn = True
        gs.my_hand = [CardEntry(name="Blue-Eyes White Dragon", code=89631139)]
        sa = [
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h1",
                card_code=89631139,
                location=LOCATION_HAND,
                sequence=0,
                sub_action="s",
            ),
            StructuredAction(category=IDLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"], structured_actions=sa)
        return builder.build(gs, prompt)

    def _battle() -> dict:
        gs = MUDGameState()
        gs.my_lp, gs.opp_lp = 8000, 6000
        gs.turn = 2
        gs.phase = "battle phase"
        gs.is_my_turn = True
        gs.my_mzone = [
            CardEntry(name="Blue-Eyes White Dragon", code=89631139, position="face-up attack")
        ]
        sa = [
            StructuredAction(
                category=BATTLE_ATTACK,
                cardspec="m1",
                card_code=89631139,
                location=LOCATION_MZONE,
                sequence=0,
                sub_action="m1",
            ),
            StructuredAction(category=BATTLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(
            prompt_type=PromptType.BATTLE_MENU, options=["a", "e"], structured_actions=sa
        )
        return builder.build(gs, prompt)

    return {"idle": _idle, "battle": _battle}
