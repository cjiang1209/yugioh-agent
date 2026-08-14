"""Tests for cli/play_client.py:display_state's global-state rendering.

YuGiOhObservation carries a structured obs.global_state (a GlobalState), and
display_state reads its fields directly -- there is no buffer layout to get
wrong. What's worth pinning is that display_state reads the RIGHT field under
the right name (a copy-paste could easily print my_hand where opp_hand
belongs) and that it renders raw, unclamped values faithfully rather than
assuming some byte-width ceiling.
"""

from __future__ import annotations

from cli.play_client import display_state

from yugioh_core.constants import PHASE_END, PHASE_MAIN2
from yugioh_env.action_describer import ActionDescriber
from yugioh_env.models import GlobalState, YuGiOhObservation

# No active prompt and no legal actions, so ActionDescriber never touches its
# card_db -- a stand-in is safe here.
_DESCRIBER = ActionDescriber(None, sys_strings=None)


def _render(capsys, **fields) -> str:
    """Render an observation carrying only the GlobalState under test: no
    active prompt and no legal actions."""
    obs = YuGiOhObservation(global_state=GlobalState(**fields))
    display_state(obs, step_num=0, action_describer=_DESCRIBER)
    return capsys.readouterr().out


def test_display_state_prints_every_global_field(capsys):
    """Every field gets a DISTINCT value, so a copy-paste that reads the wrong
    attribute (e.g. opp_hand where my_hand belongs) shows up as a wrong number
    in the output rather than coincidentally matching."""
    out = _render(
        capsys,
        my_lp=8000,
        opp_lp=7000,
        turn=5,
        phase=PHASE_MAIN2,
        is_my_turn=True,
        chain_count=2,
        my_deck=30,
        my_hand=6,
        my_grave=3,
        my_banished=1,
        my_extra=9,
        opp_deck=28,
        opp_hand=7,
        opp_grave=4,
        opp_banished=2,
        opp_extra=8,
    )

    assert "Turn 5" in out
    assert "<-- YOUR TURN" in out
    assert "YOUR LP:  8000" in out
    assert "OPP LP:  7000" in out
    assert "Hand:  6" in out
    assert "Deck: 30" in out
    assert "GY:  3" in out
    assert "Ban:  1" in out
    assert "Extra:  9" in out
    assert "Opp Hand:  7" in out
    assert "Chain count: 2" in out


def test_display_state_phase_name_keeps_its_high_byte(capsys):
    """MAIN2 (0x100) and END (0x200) live entirely in the high byte of the
    engine bitmask. GlobalState.phase carries the raw int, so this pins that
    display_state's PHASE_NAMES lookup resolves the wide value and that a
    false turn marker doesn't leak through."""
    out = _render(capsys, phase=PHASE_END, is_my_turn=False)

    assert "Phase: End" in out
    assert "<-- YOUR TURN" not in out


def test_display_state_omits_chain_line_when_no_chain(capsys):
    out = _render(capsys, chain_count=0)

    assert "Chain count" not in out


def test_display_state_reports_lp_above_the_byte_ceiling(capsys):
    """The byte path clamps LP to a uint16 (max 65535). GlobalState.my_lp is
    a plain int with no such ceiling, so display_state must show the raw
    engine value rather than a wrapped or clamped one."""
    out = _render(capsys, my_lp=99999, opp_lp=8000)

    assert "YOUR LP: 99999" in out
