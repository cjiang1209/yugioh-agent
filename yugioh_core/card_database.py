"""SQLite reader for .cdb card database files."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

from yugioh_core.constants import TYPE_LINK, split_setcodes


class CardDatabase:
    """Reads card data from a .cdb SQLite database.

    The database has two main tables:
    - datas: code, ot, alias, setcode, type, atk, def, level, race, attribute
    - texts: code, name, desc, str1..str16
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # cards.cdb is immutable per-process; cache name lookups to avoid
        # N+1 SQLite roundtrips from describer hot paths.
        self._name_cache: dict[int, str] = {}
        self._cache: dict[int, dict | None] = {}
        self._desc_cache: dict[int, str] = {}

    def get_card(self, code: int) -> dict | None:
        """Get card data by passcode. Returns None if not found.

        Reports what cards.cdb stores, which is what the engine needs — not what
        the card prints. Spell/Trap rows can carry real `race` and `attribute`
        values, for instance, because those cards become monsters at runtime;
        display consumers decide for themselves what to show.

        The one value normalized here is `defense`, which is None for Link
        monsters: cards.cdb reuses that column for the arrow bitmask, so a Link
        monster has no defense to report at all.
        """
        if code in self._cache:
            return self._cache[code]

        cursor = self._conn.execute("SELECT * FROM datas WHERE id=?", (code,))
        row = cursor.fetchone()
        if row is None:
            self._cache[code] = None
            return None

        # Parse level field: contains level, lscale, rscale packed
        level_raw = row["level"]
        level = level_raw & 0xFF
        lscale = (level_raw >> 24) & 0xFF
        rscale = (level_raw >> 16) & 0xFF

        setcodes = split_setcodes(row["setcode"])

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

        if card["type"] & TYPE_LINK:
            card["link_marker"] = row["def"]
            card["defense"] = None

        self._cache[code] = card
        return card

    def get_card_names_batch(self, codes: Iterable[int]) -> dict[int, str]:
        """Get card names for multiple passcodes in a single query."""
        unique = list(set(codes))
        if not unique:
            return {}
        placeholders = ",".join("?" * len(unique))
        rows = self._conn.execute(
            f"SELECT id, name FROM texts WHERE id IN ({placeholders})", unique
        ).fetchall()
        return {r["id"]: r["name"] for r in rows}

    def get_card_name(self, code: int) -> str:
        """Get card name by passcode."""
        cached = self._name_cache.get(code)
        if cached is not None:
            return cached
        cursor = self._conn.execute("SELECT name FROM texts WHERE id=?", (code,))
        row = cursor.fetchone()
        name = row["name"] if row else f"Unknown({code})"
        self._name_cache[code] = name
        return name

    def get_card_string(self, code: int, n: int) -> str | None:
        """Look up card-specific string `n` (0-15) for the given passcode.

        These are the per-card option strings the engine references via
        `aux.Stringid(code, n)` — stored as `texts.str{n+1}` in cards.cdb.
        Returns None when the card or slot is unknown, or the string is empty.
        """
        if not 0 <= n <= 15:
            return None
        column = f"str{n + 1}"
        cursor = self._conn.execute(f"SELECT {column} FROM texts WHERE id=?", (code,))
        row = cursor.fetchone()
        if row is None:
            return None
        value = row[column]
        return value if value else None

    def get_card_desc(self, code: int) -> str | None:
        """Look up the card's rules text (`texts.desc`), or None when absent.

        cards.cdb stores CRLF line endings; they are normalized to "\\n" so a
        single convention reaches the API, the DOM and test assertions. The
        breaks themselves are meaningful — many descriptions are multi-line, and
        Pendulum cards carry "[ Pendulum Effect ]" and "[ Monster Effect ]"
        sections split by a divider — so nothing else about the text is altered.
        """
        cached = self._desc_cache.get(code)
        if cached is not None:
            return cached
        cursor = self._conn.execute('SELECT "desc" FROM texts WHERE id=?', (code,))
        row = cursor.fetchone()
        if row is None:
            return None
        raw = row["desc"]
        if not raw:
            return None
        desc = raw.replace("\r\n", "\n").replace("\r", "\n")
        self._desc_cache[code] = desc
        return desc

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self._conn.close()
