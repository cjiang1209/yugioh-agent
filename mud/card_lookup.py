"""Reverse index from card name to passcode.

Resolves card names from MUD text (e.g. "Blue-Eyes White Dragon") to
passcodes (e.g. 89631139) using the ``texts`` table in cards.cdb.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class CardNameLookup:
    """Name-to-code reverse index built from cards.cdb texts table.

    Lookup is exact-match (both the MUD server and the bot use the same DB,
    so names always match verbatim).

    Alternate artwork cards share the same name but have different passcodes.
    The ``datas.alias`` field distinguishes them: canonical cards have
    ``alias=0``, alternates point to the canonical ID.  This class prefers
    the canonical passcode for each name.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._name_to_code: dict[str, int] = {}
        with sqlite3.connect(str(db_path)) as conn:
            # ORDER BY alias ASC puts canonical cards (alias=0) first.
            # We skip names already seen, so the canonical code wins.
            rows = conn.execute(
                "SELECT t.id, t.name, COALESCE(d.alias, 0) AS alias"
                " FROM texts t"
                " LEFT JOIN datas d ON t.id = d.id"
                " ORDER BY alias ASC"
            )
            for code, name, _alias in rows:
                if name and name not in self._name_to_code:
                    self._name_to_code[name] = code

    def name_to_code(self, name: str) -> int | None:
        """Return the passcode for *name*, or ``None`` if not found."""
        return self._name_to_code.get(name)

    def __len__(self) -> int:
        return len(self._name_to_code)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_code
