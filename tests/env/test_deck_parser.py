"""Test .ydk deck file parser."""

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


def test_inline_comment_stripped(tmp_path):
    """A `# name` suffix after the card code should be treated as a comment."""
    ydk = tmp_path / "inline.ydk"
    ydk.write_text(
        "#main\n89631139 # Blue-Eyes White Dragon\n46986414  # Dark Magician\n#extra\n!side\n"
    )
    deck = parse_ydk(ydk)
    assert deck["main"] == [89631139, 46986414]
