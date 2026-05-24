"""Tests for cli/play_client.py:_format_prompt_summary.

Covers each prompt-type rendering branch: the few that surface
constraints/progress (select_card, tribute, chain_link forced) and
the fallback path that returns just the type label.
"""

from cli.play_client import _format_prompt_summary


def test_none_prompt_returns_empty_string():
    assert _format_prompt_summary(None) == ""


def test_empty_prompt_returns_empty_string():
    assert _format_prompt_summary({}) == ""


def test_unknown_type_falls_through_to_label():
    assert _format_prompt_summary({"type": "position"}) == "Position"
    assert _format_prompt_summary({"type": "place"}) == "Place"
    assert _format_prompt_summary({"type": "rps"}) == "Rock-Paper-Scissors"


def test_yes_no_without_prompt_text_returns_type_label():
    """When prompt_text is absent or null, fall back to today's behavior."""
    assert _format_prompt_summary({"type": "yes_no"}) == "Yes/No"
    assert _format_prompt_summary({"type": "yes_no", "prompt_text": None}) == "Yes/No"


def test_yes_no_with_prompt_text_appends_resolved_question():
    summary = _format_prompt_summary(
        {
            "type": "yes_no",
            "prompt_text": "Pay 1000 LP?",
        }
    )
    assert summary == "Yes/No — Pay 1000 LP?"


def test_effect_yn_with_prompt_text_appends_resolved_question():
    summary = _format_prompt_summary(
        {
            "type": "effect_yn",
            "prompt_text": 'Activate the Trigger Effect of "Blue-Eyes" from [Monster Zone]?',
        }
    )
    assert (
        summary == 'Effect Yes/No — Activate the Trigger Effect of "Blue-Eyes" from [Monster Zone]?'
    )


def test_truly_unknown_type_returns_raw_string():
    assert _format_prompt_summary({"type": "future_kind"}) == "future_kind"


def test_select_card_fixed_count():
    """min == max → 'pick N' (no range)."""
    summary = _format_prompt_summary(
        {
            "type": "select_card",
            "min": 1,
            "max": 1,
            "selected_count": 0,
        }
    )
    assert summary == "Select Card — pick 1"


def test_select_card_range():
    summary = _format_prompt_summary(
        {
            "type": "select_card",
            "min": 1,
            "max": 3,
            "selected_count": 0,
        }
    )
    assert summary == "Select Card — pick 1 to 3"


def test_select_card_with_progress():
    summary = _format_prompt_summary(
        {
            "type": "select_card",
            "min": 2,
            "max": 2,
            "selected_count": 1,
        }
    )
    assert summary == "Select Card — pick 2, 1 selected"


def test_select_card_finishable_appended():
    """MSG_SELECT_UNSELECT_CARD maps to type='select_card' but emits
    `finishable` instead of `selected_count`. The summary surfaces
    finishable as a constraint hint."""
    summary = _format_prompt_summary(
        {
            "type": "select_card",
            "min": 1,
            "max": 3,
            "finishable": True,
        }
    )
    assert summary == "Select Card — pick 1 to 3, finishable"


def test_tribute_no_progress():
    summary = _format_prompt_summary(
        {
            "type": "tribute",
            "min_release": 2,
            "max_cards": 2,
            "release_total": 0,
            "cards_selected": 0,
        }
    )
    assert summary == "Tribute — release total ≥ 2 (max 2 cards)"


def test_tribute_with_progress_singular():
    summary = _format_prompt_summary(
        {
            "type": "tribute",
            "min_release": 2,
            "max_cards": 2,
            "release_total": 1,
            "cards_selected": 1,
        }
    )
    assert summary == "Tribute — release total ≥ 2 (max 2 cards), release=1/2 (1 card)"


def test_tribute_with_progress_plural():
    summary = _format_prompt_summary(
        {
            "type": "tribute",
            "min_release": 3,
            "max_cards": 3,
            "release_total": 2,
            "cards_selected": 2,
        }
    )
    assert summary == "Tribute — release total ≥ 3 (max 3 cards), release=2/3 (2 cards)"


def test_chain_link_forced():
    summary = _format_prompt_summary({"type": "chain_link", "forced": True})
    assert summary == "Chain Link — forced"


def test_chain_link_not_forced():
    summary = _format_prompt_summary({"type": "chain_link", "forced": False})
    assert summary == "Chain Link"


def test_chain_link_forced_field_absent():
    summary = _format_prompt_summary({"type": "chain_link"})
    assert summary == "Chain Link"


def test_label_dict_covers_every_describer_prompt_type():
    """Drift guard: every value emitted by the describer's _PROMPT_TYPE_MAP
    must have a display label in _PROMPT_TYPE_LABELS, otherwise the CLI
    falls through to the raw internal string instead of a friendly label.
    """
    from cli.play_client import _PROMPT_TYPE_LABELS

    from yugioh_env.action_describer import _PROMPT_TYPE_MAP

    describer_types = set(_PROMPT_TYPE_MAP.values())
    label_keys = set(_PROMPT_TYPE_LABELS)
    missing = describer_types - label_keys
    assert not missing, (
        f"_PROMPT_TYPE_LABELS missing entries for: {sorted(missing)}. "
        f"Update cli/play_client.py:_PROMPT_TYPE_LABELS to keep CLI labels "
        f"in sync with the describer's prompt types."
    )
