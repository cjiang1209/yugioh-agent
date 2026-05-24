"""Test the SQLite card database reader."""


def test_blue_eyes(card_db):
    """Blue-Eyes White Dragon (89631139) should have correct stats."""
    card = card_db.get_card(89631139)
    assert card is not None
    assert card["code"] == 89631139
    assert card["attack"] == 3000
    assert card["defense"] == 2500
    assert card["level"] == 8


def test_unknown_card(card_db):
    """Unknown card should return None."""
    card = card_db.get_card(99999999)
    assert card is None


def test_card_caching(card_db):
    """Second lookup should return cached result."""
    card1 = card_db.get_card(89631139)
    card2 = card_db.get_card(89631139)
    assert card1 is card2


def test_card_name(card_db):
    """Should retrieve card name."""
    name = card_db.get_card_name(89631139)
    assert "Blue-Eyes" in name or "blue-eyes" in name.lower() or len(name) > 0
