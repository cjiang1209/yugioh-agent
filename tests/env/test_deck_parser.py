"""Test .ydk deck file parser."""

import pytest

from yugioh_env.deck_parser import parse_ydk


def test_parse_default_deck(deck_path):
    """Default deck (blue_eyes) should have 40 main + 11 extra deck cards."""
    deck = parse_ydk(deck_path)
    assert "main" in deck
    assert "extra" in deck
    assert "side" in deck
    assert len(deck["main"]) == 40
    assert len(deck["extra"]) == 11
    assert len(deck["side"]) == 0


def test_card_codes_are_positive(deck_path):
    """All card codes should be positive integers."""
    deck = parse_ydk(deck_path)
    for code in deck["main"]:
        assert code > 0
        assert isinstance(code, int)


def test_blue_eyes_in_deck(deck_path):
    """Starter deck should contain Blue-Eyes White Dragon."""
    deck = parse_ydk(deck_path)
    assert 89631139 in deck["main"]
