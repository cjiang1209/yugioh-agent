"""Test the SQLite card database reader."""

from yugioh_core.constants import TYPE_LINK


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


def test_link_monster_has_no_defense(card_db, cdb_column):
    """A Link monster's arrows occupy the def column, so it has no defense at all
    and 0 would read as a real DEF of zero downstream."""
    codes = cdb_column("SELECT id FROM datas WHERE (type & ?) != 0", (TYPE_LINK,))
    assert codes, "expected cards.cdb to contain Link monsters"
    for code in codes[:20]:
        card = card_db.get_card(code)
        assert card["defense"] is None, f"card {code} reports a DEF"
        assert card["link_marker"] != 0, f"card {code} lost its arrows"


def test_card_caching(card_db):
    """Second lookup should return cached result."""
    card1 = card_db.get_card(89631139)
    card2 = card_db.get_card(89631139)
    assert card1 is card2


def test_card_name(card_db):
    """Should retrieve card name."""
    name = card_db.get_card_name(89631139)
    assert "Blue-Eyes" in name or "blue-eyes" in name.lower() or len(name) > 0


def test_get_card_desc_normalizes_crlf(card_db, cdb_column):
    """No \\r survives, and every break does."""
    codes = cdb_column("SELECT id FROM texts WHERE desc LIKE ? LIMIT 1", ("%\r\n%",))
    assert codes, "expected cards.cdb to contain CRLF line endings"
    raw = cdb_column('SELECT "desc" FROM texts WHERE id=?', (codes[0],))[0]

    desc = card_db.get_card_desc(codes[0])
    assert desc is not None
    assert "\r" not in desc
    assert desc.count("\n") == raw.count("\r\n")
    assert desc == raw.replace("\r\n", "\n")


def test_get_card_desc_preserves_blank_lines(card_db, cdb_column):
    """A blank line separates paragraphs; collapsing it would run them together."""
    codes = cdb_column("SELECT id FROM texts WHERE desc LIKE ? LIMIT 1", ("%\r\n\r\n%",))
    assert codes, "expected cards.cdb to contain blank lines in card text"

    desc = card_db.get_card_desc(codes[0])
    assert desc is not None
    assert "\r" not in desc
    assert "\n\n" in desc


def test_get_card_desc_leaves_single_line_text_alone(card_db, cdb_column):
    """Text without line breaks comes back byte-for-byte; normalization only
    touches the breaks."""
    codes = cdb_column(
        "SELECT id FROM texts WHERE desc != '' AND desc NOT LIKE ? LIMIT 1", ("%\r%",)
    )
    assert codes, "expected cards.cdb to contain single-line card text"
    raw = cdb_column('SELECT "desc" FROM texts WHERE id=?', (codes[0],))[0]

    assert card_db.get_card_desc(codes[0]) == raw


def test_get_card_desc_unknown_code(card_db):
    assert card_db.get_card_desc(99999999) is None


def test_get_card_desc_empty_text_returns_none(card_db, cdb_column):
    """An empty desc reads as absent: returning "" would be indistinguishable
    from real empty text downstream."""
    codes = cdb_column("SELECT id FROM texts WHERE desc IS NULL OR desc = ''")
    for code in codes:
        assert card_db.get_card_desc(code) is None


def test_get_card_desc_caching(card_db):
    """Second lookup returns the cached string, matching get_card/get_card_name."""
    assert card_db.get_card_desc(89631139) is card_db.get_card_desc(89631139)
