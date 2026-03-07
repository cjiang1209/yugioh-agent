"""SQLite reader for .cdb card database files."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class CardDatabase:
    """Reads card data from a .cdb SQLite database.

    The database has two main tables:
    - datas: code, ot, alias, setcode, type, atk, def, level, race, attribute
    - texts: code, name, desc, str1..str16
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._cache: dict[int, dict | None] = {}

    def get_card(self, code: int) -> dict | None:
        """Get card data by passcode. Returns None if not found."""
        if code in self._cache:
            return self._cache[code]

        cursor = self._conn.execute(
            "SELECT * FROM datas WHERE id=?", (code,)
        )
        row = cursor.fetchone()
        if row is None:
            self._cache[code] = None
            return None

        # Parse level field: contains level, lscale, rscale packed
        level_raw = row["level"]
        level = level_raw & 0xFF
        lscale = (level_raw >> 24) & 0xFF
        rscale = (level_raw >> 16) & 0xFF

        # Parse setcode as list of 16-bit values
        setcode_raw = row["setcode"]
        setcodes = []
        val = setcode_raw
        while val:
            sc = val & 0xFFFF
            if sc:
                setcodes.append(sc)
            val >>= 16

        card = {
            "code": row["id"],
            "alias": row["alias"],
            "setcodes": setcodes,
            "type": row["type"],
            "level": level,
            "attribute": row["attribute"],
            "race": row["race"],
            "attack": row["atk"],
            "defense": row["def"],
            "lscale": lscale,
            "rscale": rscale,
            "link_marker": 0,  # link_marker stored in def for Link monsters
        }

        # For Link monsters, defense field stores link marker
        from yugioh_env.constants import TYPE_LINK
        if card["type"] & TYPE_LINK:
            card["link_marker"] = row["def"]
            card["defense"] = 0

        self._cache[code] = card
        return card

    def get_card_name(self, code: int) -> str:
        """Get card name by passcode."""
        cursor = self._conn.execute(
            "SELECT name FROM texts WHERE id=?", (code,)
        )
        row = cursor.fetchone()
        return row["name"] if row else f"Unknown({code})"

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
