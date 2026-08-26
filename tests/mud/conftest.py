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
    from yugioh_mud.card_lookup import CardNameLookup
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
    lookup = CardNameLookup(db_path)

    def _state(turn, phase, *, lp=8000, opp_lp=8000, my_turn=True, with_lookup=False):
        """A board with the four scalars every case has to set."""
        gs = MUDGameState(card_lookup=lookup if with_lookup else None)
        gs.my_lp, gs.opp_lp = lp, opp_lp
        gs.turn, gs.phase, gs.is_my_turn = turn, phase, my_turn
        return gs

    def _idle() -> dict:
        gs = _state(1, "main1 phase")
        gs.my_hand = [
            CardEntry(name="Blue-Eyes White Dragon", code=89631139),
            CardEntry(name="Mirror Force", code=44095762),
        ]
        # Two same-category rows at different zone slots, so `index` and
        # `sequence` are both non-zero on the second -- the fields a
        # single-action case leaves at 0.
        sa = [
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h1",
                card_code=89631139,
                location=LOCATION_HAND,
                sequence=0,
                sub_action="s",
            ),
            StructuredAction(
                category=IDLE_SUMMON,
                cardspec="h2",
                card_code=44095762,
                location=LOCATION_HAND,
                sequence=3,
                sub_action="s",
            ),
            StructuredAction(category=IDLE_TO_EP, sub_action="e"),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.IDLE_CMD, options=["e"], structured_actions=sa)
        return builder.build(gs, prompt)

    def _battle() -> dict:
        gs = _state(2, "battle phase", opp_lp=6000)
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

    def _effectyn() -> dict:
        """`_encode_binary_choice`: yes/no, with the card named in the text."""
        gs = _state(3, "main1 phase", with_lookup=True)
        gs.my_mzone = [
            CardEntry(name="Blue-Eyes White Dragon", code=89631139, position="face-up attack")
        ]
        prompt = ParsedPrompt(
            prompt_type=PromptType.SELECT_EFFECTYN,
            options=["y", "n"],
            raw_lines=["Do you want to use the effect from Blue-Eyes White Dragon?"],
        )
        return builder.build(gs, prompt)

    def _select_card() -> dict:
        """`_encode_option_actions`: a pick list, three choices."""
        gs = _state(4, "main1 phase", lp=7000)
        gs.my_hand = [
            CardEntry(name="Blue-Eyes White Dragon", code=89631139),
            CardEntry(name="Mirror Force", code=44095762),
        ]
        prompt = ParsedPrompt(prompt_type=PromptType.SELECT_CARD, options=["1", "2", "3"])
        return builder.build(gs, prompt)

    def _select_place() -> dict:
        """`_encode_generic_options`: the fall-through path, index only."""
        gs = _state(5, "main2 phase", opp_lp=5000, my_turn=False)
        prompt = ParsedPrompt(prompt_type=PromptType.SELECT_PLACE, options=["m1", "m2"])
        return builder.build(gs, prompt)

    return {
        "idle": _idle,
        "battle": _battle,
        "effectyn": _effectyn,
        "select_card": _select_card,
        "select_place": _select_place,
    }
